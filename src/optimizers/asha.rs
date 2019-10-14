//! **A**synchronous **S**uccessive **H**alving **A**lgorithm.
//!
//! # References
//!
//! - [Massively Parallel Hyperparameter Tuning](https://arxiv.org/abs/1810.05934)
use crate::{
    Budget, ErrorKind, IdGen, MfObs, MultiFidelityOptimizer, Obs, ObsId, Optimizer, Ranked, Result,
};
use rand::Rng;
use std::cmp;
use std::collections::HashMap;

/// Builder of `AshaOptimizer`.
#[derive(Debug, Clone)]
pub struct AshaOptimizerBuilder {
    reduction_factor: usize,
    without_checkpoint: bool,
}
impl AshaOptimizerBuilder {
    /// Makes a new `AshaOptimizerBuilder` instance with the default settings.
    pub const fn new() -> Self {
        Self {
            reduction_factor: 2,
            without_checkpoint: false,
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

    /// Makes the resulting optimizer work well with evaluators that don't have the capability of checkpointing.
    pub fn without_checkpoint(&mut self) -> &mut Self {
        self.without_checkpoint = true;
        self
    }

    /// Builds a new `AshaOptimizer` instance.
    pub fn finish<V, O>(
        &self,
        inner: O,
        min_budget: u64,
        max_budget: u64,
    ) -> Result<AshaOptimizer<V, O>>
    where
        V: Ord,
        O: Optimizer<Value = Ranked<V>>,
    {
        track_assert!(min_budget <= max_budget, ErrorKind::InvalidInput; min_budget, max_budget);
        track_assert!(0 < min_budget, ErrorKind::InvalidInput; min_budget, max_budget);

        let rungs = Rungs::new(min_budget, max_budget, self);
        Ok(AshaOptimizer {
            inner,
            rungs,
            initial_budget: Budget::new(min_budget),
            without_checkpoint: self.without_checkpoint,
            max_budget,
        })
    }
}
impl Default for AshaOptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// [ASHA] based optimizer.
///
/// [ASHA]: https://arxiv.org/abs/1810.05934
#[derive(Debug)]
pub struct AshaOptimizer<V, O: Optimizer> {
    inner: O,
    rungs: Rungs<O::Param, V>,
    initial_budget: Budget,
    without_checkpoint: bool,
    max_budget: u64,
}
impl<V, O> AshaOptimizer<V, O>
where
    V: Ord,
    O: Optimizer<Value = Ranked<V>>,
{
    /// Makes a new `AshaOptimizer` instance with the default settings.
    pub fn new(inner: O, min_budget: u64, max_budget: u64) -> Result<Self> {
        track!(AshaOptimizerBuilder::new().finish(inner, min_budget, max_budget))
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
impl<V, O> MultiFidelityOptimizer for AshaOptimizer<V, O>
where
    V: Ord + Clone,
    O: Optimizer<Value = Ranked<V>>,
    O::Param: Clone,
{
    type Param = O::Param;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: R, mut idg: G) -> Result<MfObs<Self::Param>> {
        if let Some(mut obs) = self.rungs.ask_promotable() {
            if self.without_checkpoint {
                obs.id = track!(idg.generate())?;
                obs.budget.consumption = 0;
            }
            Ok(obs)
        } else {
            let obs = track!(self.inner.ask(rng, idg))?;
            let obs = MfObs::from_obs(obs, self.initial_budget);
            Ok(obs)
        }
    }

    fn tell(&mut self, obs: MfObs<Self::Param, Self::Value>) -> Result<()> {
        track_assert!(
            obs.budget.consumption <= self.max_budget,
            ErrorKind::InvalidInput; obs.id, obs.budget, self.max_budget
        );

        if obs.budget.consumption < obs.budget.amount {
            // The evaluation of this observation was canceled.
        } else {
            track!(self.rungs.tell(obs.clone()))?;
        }

        let rank = self.max_budget - obs.budget.consumption;
        let obs = Obs::from(obs).map_value(|value| Ranked { rank, value });
        track!(self.inner.tell(obs))?;

        Ok(())
    }
}

#[derive(Debug)]
struct Rungs<P, V>(Vec<Rung<P, V>>);
impl<P, V> Rungs<P, V>
where
    V: Ord,
{
    fn new(min_budget: u64, max_budget: u64, builder: &AshaOptimizerBuilder) -> Self {
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

    fn ask_promotable(&mut self) -> Option<MfObs<P>> {
        for rung in self.0.iter_mut().rev() {
            if let Some(obs) = rung.ask_promotable() {
                return Some(obs);
            }
        }
        None
    }

    fn tell(&mut self, obs: MfObs<P, V>) -> Result<()> {
        use std::u64;

        for rung in self.0.iter_mut().rev() {
            let p = obs.budget.consumption;
            if rung.curr_budget <= p && p < rung.next_budget.unwrap_or(u64::MAX) {
                track!(rung.tell(obs))?;
                return Ok(());
            }
        }
        track_panic!(ErrorKind::InvalidInput; obs.id);
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
    fn new(curr_budget: u64, next_budget: Option<u64>, builder: &AshaOptimizerBuilder) -> Self {
        Self {
            obss: HashMap::new(),
            curr_budget,
            next_budget,
            reduction_factor: builder.reduction_factor,
        }
    }

    fn ask_promotable(&mut self) -> Option<MfObs<P>> {
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
            let (mut obs, value) = if let Config::Pending { obs } =
                self.obss.remove(&id).unwrap_or_else(|| unreachable!())
            {
                obs.take_value()
            } else {
                unreachable!()
            };

            self.obss.insert(id, Config::Finished { value });

            obs.budget.amount = next_budget;
            Some(obs)
        } else {
            None
        }
    }

    fn tell(&mut self, obs: MfObs<P, V>) -> Result<()> {
        track_assert!(!self.obss.contains_key(&obs.id), ErrorKind::Bug);
        track_assert!(
            self.curr_budget <= obs.budget.consumption,
            ErrorKind::InvalidInput; self.curr_budget, obs.budget
        );
        self.obss.insert(obs.id, Config::Pending { obs });
        Ok(())
    }
}

#[derive(Debug)]
enum Config<P, V> {
    Pending { obs: MfObs<P, V> },
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
    use crate::domains::ContinuousDomain;
    use crate::generators::SerialIdGenerator;
    use crate::optimizers::random::RandomOptimizer;
    use rand;
    use trackable::result::TestResult;

    #[test]
    fn asha_works() -> TestResult {
        let inner = RandomOptimizer::new(track!(ContinuousDomain::new(0.0, 1.0))?);
        let mut optimizer = track!(AshaOptimizer::<usize, _>::new(inner, 10, 20))?;

        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        // first
        let obs = track!(optimizer.ask(&mut rng, &mut idg))?;
        assert_eq!(obs.id.get(), 0);

        let mut obs = obs.map_value(|_| 1);
        obs.budget.consumption += 10;
        track!(optimizer.tell(obs))?;

        // second
        let obs = track!(optimizer.ask(&mut rng, &mut idg))?;
        assert_eq!(obs.id.get(), 1);

        let mut obs = obs.map_value(|_| 2);
        obs.budget.consumption += 10;
        track!(optimizer.tell(obs))?;

        // third
        let obs = track!(optimizer.ask(&mut rng, &mut idg))?;
        assert_eq!(obs.id.get(), 0);

        let mut obs = obs.map_value(|_| 1);
        obs.budget.consumption += 10;
        track!(optimizer.tell(obs))?;

        Ok(())
    }
}
