use super::parzen_estimator::ParzenEstimatorBuilder;
use super::{DefaultPreprocessor, Preprocess, TpeOptions};
use crate::float::NonNanF64;
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::spaces::Numerical;
use crate::Result;
use rand::distributions::Distribution;
use rand::Rng;
use std::collections::HashMap;

/// TPE optimizer for numerical parameter.
#[derive(Debug)]
pub struct TpeNumericalOptimizer<P: Numerical, V, T = DefaultPreprocessor> {
    param_space: P,
    options: TpeOptions<T>,
    observations: HashMap<ObsId, Obs<f64, V>>,
    estimator_builder: ParzenEstimatorBuilder,
}
impl<P, V> TpeNumericalOptimizer<P, V, DefaultPreprocessor>
where
    P: Numerical,
    V: Ord,
{
    /// Make a new `TpeNumericalOptimizer` instance.
    pub fn new(param_space: P) -> Self {
        Self::with_options(param_space, TpeOptions::default())
    }
}
impl<P, V, T> TpeNumericalOptimizer<P, V, T>
where
    P: Numerical,
    V: Ord,
    T: Preprocess<V>,
{
    /// Make a new `TpeNumericalOptimizer` instance with the given options.
    pub fn with_options(param_space: P, options: TpeOptions<T>) -> Self {
        Self {
            param_space,
            estimator_builder: ParzenEstimatorBuilder::new(
                options.prior_weight,
                options.prior_uniform,
                options.uniform_sigma,
                options.uniform_weight,
            ),
            options,
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
impl<P, V, T> Optimizer for TpeNumericalOptimizer<P, V, T>
where
    P: Numerical,
    V: Ord,
    T: Preprocess<V>,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        let mut observations = self.observations.values().collect::<Vec<_>>();
        observations.sort_by_key(|o| &o.value);

        let gamma = self.options.preprocessor.divide_observations(&observations);
        let (superiors, inferiors) = observations.split_at(gamma);

        let superior_weights = self
            .options
            .preprocessor
            .weight_observations(superiors, true);
        let inferior_weights = self
            .options
            .preprocessor
            .weight_observations(inferiors, false);

        let superior_estimator = self.estimator_builder.finish(
            superiors.iter().map(|o| o.param),
            superior_weights,
            self.param_space.range().low,
            self.param_space.range().high,
        );

        let inferior_estimator = self.estimator_builder.finish(
            inferiors.iter().map(|o| o.param),
            inferior_weights,
            self.param_space.range().low,
            self.param_space.range().high,
        );

        let (_, param) = superior_estimator
            .gmm()
            .sample_iter(rng)
            .take(self.options.ei_candidates.get())
            .map(|candidate| {
                let superior_log_likelihood = superior_estimator.gmm().log_pdf(candidate);
                let inferior_log_likelihood = inferior_estimator.gmm().log_pdf(candidate);
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
