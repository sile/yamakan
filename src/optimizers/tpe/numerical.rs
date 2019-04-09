use super::parzen_estimator::ParzenEstimatorBuilder;
use super::{DefaultStrategy, NumericalStrategy};
use crate::float::NonNanF64;
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::spaces::{Numerical, PriorCdf, PriorDistribution, PriorPdf};
use crate::Result;
use rand::distributions::Distribution;
use rand::Rng;
use std::collections::HashMap;

/// TPE optimizer for numerical parameter.
#[derive(Debug)]
pub struct TpeNumericalOptimizer<P: Numerical, V, S = DefaultStrategy> {
    param_space: P,
    strategy: S,
    observations: HashMap<ObsId, Obs<f64, V>>,
}
impl<P, V, S> TpeNumericalOptimizer<P, V, S>
where
    P: Numerical,
    V: Ord,
    S: NumericalStrategy<V> + Default,
{
    /// Make a new `TpeNumericalOptimizer` instance.
    pub fn new(param_space: P) -> Self {
        Self::with_strategy(param_space, S::default())
    }
}
impl<P, V, S> TpeNumericalOptimizer<P, V, S>
where
    P: Numerical,
    V: Ord,
    S: NumericalStrategy<V>,
{
    /// Make a new `TpeNumericalOptimizer` instance with the given strategy.
    pub fn with_strategy(param_space: P, strategy: S) -> Self {
        Self {
            param_space,
            strategy,
            observations: HashMap::new(),
        }
    }

    /// Returns a reference to the parameter space.
    pub fn param_space(&self) -> &P {
        &self.param_space
    }

    /// Returns a mutable reference to the parameter space.
    pub fn param_space_mut(&mut self) -> &mut P {
        &mut self.param_space
    }
}
impl<P, V, S> Optimizer for TpeNumericalOptimizer<P, V, S>
where
    P: Numerical + PriorDistribution + PriorCdf + PriorPdf,
    V: Ord,
    S: NumericalStrategy<V>,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        let mut observations = self.observations.values().collect::<Vec<_>>();
        observations.sort_by_key(|o| &o.value);

        let gamma = self.strategy.division_position(&observations);
        let (superiors, inferiors) = observations.split_at(gamma);

        let superior_weights = self.strategy.superior_weights(superiors);
        let inferior_weights = self.strategy.inferior_weights(inferiors);

        let prior_weight = self.strategy.prior_weight(&observations);
        let builder = ParzenEstimatorBuilder::new(&self.param_space, &self.strategy, prior_weight);
        let superior_estimator =
            builder.finish(superiors.iter().map(|o| o.param), superior_weights);

        let inferior_estimator =
            builder.finish(inferiors.iter().map(|o| o.param), inferior_weights);

        let ei_candidates = self.strategy.ei_candidates(superiors);
        let (_, param) = superior_estimator
            .sample_iter(rng)
            .take(ei_candidates.get())
            .map(|candidate| {
                let superior_log_likelihood = superior_estimator.log_pdf(candidate);
                let inferior_log_likelihood = inferior_estimator.log_pdf(candidate);
                let ei = superior_log_likelihood - inferior_log_likelihood;
                (ei, candidate)
            })
            .max_by_key(|(ei, _)| NonNanF64::new(*ei))
            .unwrap_or_else(|| unreachable!());
        let param = track!(self.param_space.from_f64(param))?;
        track!(Obs::new(idg, param))
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        let obs = track!(obs.try_map_param(|p| self.param_space.to_f64(&p)))?;
        self.observations.insert(obs.id, obs);
        Ok(())
    }

    fn forget(&mut self, id: ObsId) -> Result<()> {
        self.observations.remove(&id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::SerialIdGenerator;
    use crate::optimizers::Optimizer;
    use crate::spaces::F64;
    use rand;
    use trackable::result::TestResult;

    #[test]
    fn tpe_numerical_works() -> TestResult {
        let param_space = track!(F64::new(0.0, 1.0))?;
        let mut opt = TpeNumericalOptimizer::<_, usize>::new(param_space);
        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.tell(obs.map_value(|_| 10)))?;

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.forget(obs.id))?;

        Ok(())
    }
}
