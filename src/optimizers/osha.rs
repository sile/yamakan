//! **O**ptimizer **S**uccessive **H**alving **A**lgorithm.
use crate::observation::{IdGen, Obs, ObsId};
use crate::{ErrorKind, Optimizer, Result};
use rand::Rng;
use std::cmp::Reverse;

/// **O**ptimizer **S**uccessive **H**alving **A**lgorithm.
#[derive(Debug)]
pub struct OshaOptimizer<M, O>
where
    O: Optimizer,
{
    meta_optimizer: M,
    active: Option<OptimizerState<O, O::Value>>,
    optimizers: Vec<OptimizerState<O, O::Value>>,
    min_evals: usize,
    warmup_evals: usize,
}
impl<M, O> OshaOptimizer<M, O>
where
    O: Optimizer,
{
    /// Makes a new `OshaOptimizer` instance.
    pub fn new(meta_optimizer: M) -> Self {
        Self {
            meta_optimizer,
            active: None,
            optimizers: Vec::new(),
            min_evals: 10,
            warmup_evals: 10,
        }
    }

    /// Makes a new `OshaOptimizer` instance with the given `min_evals`.
    pub fn with_min_evals(meta_optimizer: M, min_evals: usize) -> Self {
        Self {
            meta_optimizer,
            active: None,
            optimizers: Vec::new(),
            min_evals,
            warmup_evals: 10,
        }
    }
}
impl<M, O> OshaOptimizer<M, O>
where
    M: Optimizer<Param = Option<O>>,
    O: Optimizer<Value = M::Value>,
    M::Value: Ord + Clone,
{
    fn activate(&mut self) -> bool {
        assert!(self.active.is_none());

        for rung in 0.. {
            // TODO: optimize
            self.optimizers
                .sort_by(|a, b| a.key(rung).cmp(&b.key(rung)));

            let n = self.optimizers.iter().filter(|o| o.rung >= rung).count();
            if n == 0 {
                return true;
            }
            let i = self
                .optimizers
                .iter()
                .position(|o| o.rung == rung)
                .unwrap_or_else(|| unreachable!());

            if i < n / 2 {
                let mut optimizer = self.optimizers.swap_remove(i);
                optimizer.rung_evals *= 2;
                optimizer.rung += 1;
                if optimizer.stagnated {
                    self.optimizers.push(optimizer);
                    return false;
                } else {
                    if optimizer.evals >= self.warmup_evals {
                        optimizer.stagnated = true;
                    }
                    self.active = Some(optimizer);
                    return true;
                }
            }
        }
        unreachable!();
    }
}
impl<M, O> Optimizer for OshaOptimizer<M, O>
where
    M: Optimizer<Param = Option<O>>,
    O: Optimizer<Value = M::Value>,
    M::Value: Ord + Clone,
{
    type Param = O::Param;
    type Value = O::Value;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param>> {
        if self.active.is_none() {
            let obs = track!(self.meta_optimizer.ask(rng, idg))?;
            let optimizer = track_assert_some!(obs.param, ErrorKind::Other);
            self.active = Some(OptimizerState::new(obs.id, optimizer, self.min_evals));
        }

        let optimizer = self.active.as_mut().unwrap_or_else(|| unreachable!());
        track!(optimizer.inner.ask(rng, idg))
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        let optimizer = track_assert_some!(self.active.as_mut(), ErrorKind::Other);

        let value = obs.value.clone();
        track!(optimizer.inner.tell(obs))?;

        if optimizer.best().map_or(true, |best| value < *best) {
            optimizer.set_best(value.clone());
            optimizer.stagnated = false;
            track!(self.meta_optimizer.tell(Obs {
                id: optimizer.id,
                param: None,
                value,
            }))?;
        }
        optimizer.evals += 1;
        if optimizer.evals >= optimizer.rung_evals {
            self.optimizers
                .push(self.active.take().unwrap_or_else(|| unreachable!()));
            while !self.activate() {}
        }

        Ok(())
    }

    fn forget(&mut self, id: ObsId) -> Result<()> {
        for o in &mut self.optimizers {
            track!(o.inner.forget(id))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct OptimizerState<O, V> {
    id: ObsId,
    bests: Vec<V>,
    evals: usize,
    rung_evals: usize,
    rung: usize,
    stagnated: bool,
    inner: O,
}
impl<O, V> OptimizerState<O, V> {
    fn new(id: ObsId, inner: O, min_evals: usize) -> Self {
        Self {
            id,
            bests: Vec::new(),
            evals: 0,
            rung_evals: min_evals,
            rung: 0,
            stagnated: false,
            inner,
        }
    }

    fn best(&self) -> Option<&V> {
        self.bests.get(self.rung)
    }

    fn set_best(&mut self, value: V) {
        if self.best().is_none() {
            self.bests.push(value);
        } else {
            self.bests[self.rung] = value;
        }
    }

    fn key(&self, rung: usize) -> Reverse<Option<Reverse<&V>>> {
        Reverse(self.bests.get(rung).map(Reverse))
    }
}
