use super::parzen_estimator::ParzenEstimatorBuilder;
use super::{DefaultPreprocessor, Preprocess, TpeOptions};
use crate::float::NonNanF64;
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::space::ParamSpace;
use crate::Result;
use rand::distributions::Distribution;
use rand::Rng;
use std::collections::HashMap;

#[derive(Debug)]
pub struct TpeNumericalOptimizer<P, V, T = DefaultPreprocessor>
where
    P: ParamSpace<Internal = f64>,
{
    param_space: P,
    options: TpeOptions<T>,
    observations: HashMap<ObsId, Obs<P::External, V>>,
    estimator_builder: ParzenEstimatorBuilder,
}
impl<P, V> TpeNumericalOptimizer<P, V, DefaultPreprocessor>
where
    P: ParamSpace<Internal = f64>,
    V: Ord,
{
    pub fn new(param_space: P) -> Self {
        Self::with_options(param_space, TpeOptions::default())
    }
}
impl<P, V, T> TpeNumericalOptimizer<P, V, T>
where
    P: ParamSpace<Internal = f64>,
    V: Ord,
    T: Preprocess<P::External, V>,
{
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

    pub fn param_space(&self) -> &P {
        &self.param_space
    }
}
impl<P, V, T> Optimizer for TpeNumericalOptimizer<P, V, T>
where
    P: ParamSpace<Internal = f64>,
    V: Ord,
    T: Preprocess<P::External, V>,
{
    type Param = P::External;
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
            superiors
                .iter()
                .map(|o| self.param_space.internalize(&o.param)),
            superior_weights,
            self.param_space.range().start,
            self.param_space.range().end,
        );

        let inferior_estimator = self.estimator_builder.finish(
            inferiors
                .iter()
                .map(|o| self.param_space.internalize(&o.param)),
            inferior_weights,
            self.param_space.range().start,
            self.param_space.range().end,
        );

        let param = superior_estimator
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
            .map(|(_, internal)| self.param_space.externalize(&internal))
            .expect("never fails");
        track!(Obs::new(idg, param))
    }

    fn tell(&mut self, observation: Obs<Self::Param, Self::Value>) -> Result<()> {
        let internal_param = self.param_space.internalize(&observation.param);
        assert!(!internal_param.is_nan());

        let r = self.param_space.range();
        assert!(
            r.start <= internal_param && internal_param < r.end,
            "Out of the range: internal_param={}, low={}, high={}",
            internal_param,
            r.start,
            r.end
        );

        self.observations.insert(observation.id, observation);
        Ok(())
    }

    fn forget(&mut self, id: ObsId) -> Result<()> {
        self.observations.remove(&id);
        Ok(())
    }
}
