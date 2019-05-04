//! **A**synchronous **S**uccessive **H**alving **A**lgorithm.
//!
//! # References
//!
//! - [Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934)
use crate::budget::{Budget, Budgeted, Leveled};
use crate::observation::{IdGen, Obs, ObsId};
use crate::{ErrorKind, Optimizer, Result};
use rand::Rng;
use std::cmp;
use std::collections::HashMap;

// TODO: rename to `AshaOptimizerBuilder`
/// Builder of `AshaOptimizer`.
#[derive(Debug, Clone)]
pub struct AshaBuilder {
    reduction_factor: usize,
}
impl AshaBuilder {
    /// Makes a new `AshaBuilder` instance with the default settings.
    pub const fn new() -> Self {
        Self {
            reduction_factor: 2,
        }
    }

    /// Sets the reduction factor of the resulting optimizer.
    ///
    /// # Errors
    ///
    /// If `factor` is less than `2`, an `ErrorKind::InvalidInput` error will be returned.
    pub fn reduction_factor(&mut self, factor: usize) -> Result<&mut Self> {
        track_assert!(factor > 1, ErrorKind::InvalidInput; factor);
        self.reduction_factor = factor;
        Ok(self)
    }

    /// Builds a new `AshaOptimizer` instance.
    pub fn finish<O, V>(
        &self,
        inner: O,
        min_budget: u64,
        max_budget: u64,
    ) -> Result<AshaOptimizer<O, V>>
    where
        O: Optimizer<Value = Leveled<V>>,
        V: Ord,
    {
        track_assert!(min_budget <= max_budget, ErrorKind::InvalidInput; min_budget, max_budget);
        track_assert!(0 < min_budget, ErrorKind::InvalidInput; min_budget, max_budget);

        let rungs = Rungs::new(min_budget, max_budget, self);
        Ok(AshaOptimizer {
            inner,
            rungs,
            initial_budget: Budget::new(min_budget),
        })
    }
}
impl Default for AshaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// [ASHA] based optimizer.
///
/// [ASHA]: https://arxiv.org/abs/1810.05934
#[derive(Debug)]
pub struct AshaOptimizer<O: Optimizer, V> {
    inner: O,
    rungs: Rungs<O::Param, V>,
    initial_budget: Budget,
}
impl<O, V> AshaOptimizer<O, V>
where
    O: Optimizer<Value = Leveled<V>>,
    V: Ord,
{
    /// Makes a new `AshaOptimizer` instance with the default settings.
    pub fn new(inner: O, min_budget: u64, max_budget: u64) -> Result<Self> {
        track!(AshaBuilder::new().finish(inner, min_budget, max_budget))
    }

    /// Returns a references to the underlying optimizer.
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Returns a mutable references to the underlying optimizer.
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    /// Consumes the `AshaOptimizer`, returning the underlying optimizer.
    pub fn into_inner(self) -> O {
        self.inner
    }
}
impl<O, V> Optimizer for AshaOptimizer<O, V>
where
    O: Optimizer<Value = Leveled<V>>,
    O::Param: Clone,
    V: Ord + Clone,
{
    type Param = Budgeted<O::Param>;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param>> {
        if let Some(obs) = self.rungs.ask_promotable() {
            Ok(obs)
        } else {
            let obs = track!(self.inner.ask(rng, idg))?;
            let obs = obs.map_param(|p| Budgeted::new(self.initial_budget, p));
            Ok(obs)
        }
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        let rung = track!(self.rungs.tell(obs.clone()))?;

        let obs = obs
            .map_param(Budgeted::into_inner)
            .map_value(|v| Leveled::new(rung as u64, v));
        track!(self.inner.tell(obs))?;

        Ok(())
    }

    fn forget(&mut self, id: ObsId) -> Result<()> {
        self.rungs.forget(id);
        Ok(())
    }
}

#[derive(Debug)]
struct Rungs<P, V>(Vec<Rung<P, V>>);
impl<P, V> Rungs<P, V>
where
    V: Ord,
{
    fn new(min_budget: u64, max_budget: u64, builder: &AshaBuilder) -> Self {
        let mut rungs = Vec::new();
        let mut budget = min_budget;
        while budget < max_budget {
            let next_budget = cmp::min(
                max_budget,
                budget.saturating_mul(builder.reduction_factor as u64),
            );
            rungs.push(Rung::new(budget, Some(next_budget), builder));
            budget = next_budget;
        }
        rungs.push(Rung::new(max_budget, None, builder));
        Self(rungs)
    }

    fn ask_promotable(&mut self) -> Option<Obs<Budgeted<P>>> {
        for rung in self.0.iter_mut().rev() {
            if let Some(obs) = rung.ask_promotable() {
                return Some(obs);
            }
        }
        None
    }

    fn tell(&mut self, obs: Obs<Budgeted<P>, V>) -> Result<usize> {
        for (i, rung) in self.0.iter_mut().enumerate() {
            if !rung.obss.contains_key(&obs.id) {
                track!(rung.tell(obs))?;
                return Ok(i);
            }
        }
        track_panic!(ErrorKind::InvalidInput; obs.id);
    }

    fn forget(&mut self, id: ObsId) {
        for rung in &mut self.0 {
            rung.forget(id);
        }
    }
}

#[derive(Debug)]
struct Rung<P, V> {
    obss: HashMap<ObsId, Config<P, V>>,
    curr_budget: u64,
    next_budget: Option<u64>,
    reduction_factor: usize,
}
impl<P, V> Rung<P, V>
where
    V: Ord,
{
    fn new(curr_budget: u64, next_budget: Option<u64>, builder: &AshaBuilder) -> Self {
        Self {
            obss: HashMap::new(),
            curr_budget,
            next_budget,
            reduction_factor: builder.reduction_factor,
        }
    }

    fn ask_promotable(&mut self) -> Option<Obs<Budgeted<P>>> {
        let next_budget = if let Some(next_budget) = self.next_budget {
            next_budget
        } else {
            return None;
        };

        // FIXME: optimize
        let mut configs = self.obss.values().collect::<Vec<_>>();
        configs.sort_by_key(|c| c.value());

        let mut found = None;
        let promotables = self.obss.len() / self.reduction_factor;
        for c in configs.iter().take(promotables) {
            if let Config::Pending { obs } = c {
                found = Some(obs.id);
                break;
            }
        }

        if let Some(id) = found {
            let (mut param, value) = if let Config::Pending { obs } =
                self.obss.remove(&id).unwrap_or_else(|| unreachable!())
            {
                (obs.param, obs.value)
            } else {
                unreachable!()
            };
            self.obss.insert(id, Config::Finished { value });

            param.budget_mut().amount = next_budget;
            Some(Obs {
                id,
                param,
                value: (),
            })
        } else {
            None
        }
    }

    fn tell(&mut self, obs: Obs<Budgeted<P>, V>) -> Result<()> {
        track_assert!(!self.obss.contains_key(&obs.id), ErrorKind::Bug);
        track_assert!(
            self.curr_budget <= obs.param.budget().consumption,
            ErrorKind::InvalidInput; self.curr_budget, obs.param.budget()
        );
        self.obss.insert(obs.id, Config::Pending { obs });
        Ok(())
    }

    fn forget(&mut self, id: ObsId) {
        self.obss.remove(&id);
    }
}

#[derive(Debug)]
enum Config<P, V> {
    Pending { obs: Obs<Budgeted<P>, V> },
    Finished { value: V },
}
impl<P, V> Config<P, V> {
    fn value(&self) -> &V {
        match self {
            Config::Pending { obs } => &obs.value,
            Config::Finished { value } => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::SerialIdGenerator;
    use crate::optimizers::random::RandomOptimizer;
    use crate::parameters::F64;
    use crate::Optimizer;
    use rand;
    use trackable::result::TestResult;

    #[test]
    fn asha_works() -> TestResult {
        let inner = RandomOptimizer::new(track!(F64::new(0.0, 1.0))?);
        let mut optimizer = track!(AshaOptimizer::<_, usize>::new(inner, 10, 20))?;

        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        // first
        let obs = track!(optimizer.ask(&mut rng, &mut idg))?;
        assert_eq!(obs.id.get(), 0);

        let mut obs = obs.map_value(|_| 1);
        track!(obs.param.budget_mut().consume(10))?;
        track!(optimizer.tell(obs))?;

        // second
        let obs = track!(optimizer.ask(&mut rng, &mut idg))?;
        assert_eq!(obs.id.get(), 1);

        let mut obs = obs.map_value(|_| 2);
        track!(obs.param.budget_mut().consume(10))?;
        track!(optimizer.tell(obs))?;

        // third
        let obs = track!(optimizer.ask(&mut rng, &mut idg))?;
        assert_eq!(obs.id.get(), 0);

        let mut obs = obs.map_value(|_| 1);
        track!(obs.param.budget_mut().consume(10))?;
        track!(optimizer.tell(obs))?;

        Ok(())
    }
}
