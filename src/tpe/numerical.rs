use super::ParzenEstimatorBuilder;
use super::{DefaultTpeStrategy, Observation, TpeStrategy};
use crate::float::NonNanF64;
use crate::optimizer::Optimizer;
use crate::space::NumericalSpace;
use rand::distributions::Distribution;
use rand::Rng;

#[derive(Debug)]
pub struct TpeNumericalOptimizer<P, V, S = DefaultTpeStrategy>
where
    P: NumericalSpace,
{
    param_space: P,
    strategy: S,
    observations: Vec<Observation<P::Param, V>>,
    estimator_builder: ParzenEstimatorBuilder,
    ei_candidates: usize,
}
impl<P, V> TpeNumericalOptimizer<P, V, DefaultTpeStrategy>
where
    P: NumericalSpace,
    V: Ord,
{
    pub fn new(param_space: P) -> Self {
        Self {
            param_space,
            strategy: DefaultTpeStrategy,
            observations: Vec::new(),
            estimator_builder: ParzenEstimatorBuilder::new(),
            ei_candidates: 24,
        }
    }
}
impl<P, V, S> Optimizer for TpeNumericalOptimizer<P, V, S>
where
    P: NumericalSpace,
    V: Ord,
    S: TpeStrategy<P::Param, V>,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let (superiors, inferiors) = self.strategy.divide_observations(&self.observations);
        let superior_weights = self.strategy.weight_superiors(superiors);
        let inferior_weights = self.strategy.weight_superiors(inferiors);
        assert_eq!(superiors.len(), superior_weights.len());
        assert_eq!(inferiors.len(), inferior_weights.len());

        let superior_estimator = self.estimator_builder.finish(
            superiors
                .iter()
                .map(|o| self.param_space.param_to_internal(&o.param)),
            superior_weights.into_iter(),
            self.param_space.internal_range().start,
            self.param_space.internal_range().end,
        );

        let inferior_estimator = self.estimator_builder.finish(
            inferiors
                .iter()
                .map(|o| self.param_space.param_to_internal(&o.param)),
            inferior_weights.into_iter(),
            self.param_space.internal_range().start,
            self.param_space.internal_range().end,
        );

        superior_estimator
            .gmm()
            .sample_iter(rng)
            .take(self.ei_candidates)
            .map(|candidate| {
                let superior_log_likelihood = superior_estimator.gmm().log_pdf(candidate);
                let inferior_log_likelihood = inferior_estimator.gmm().log_pdf(candidate);
                let ei = superior_log_likelihood - inferior_log_likelihood;
                (ei, candidate)
            })
            .max_by_key(|(ei, _)| NonNanF64::new(*ei))
            .map(|(_, internal)| self.param_space.internal_to_param(internal))
            .expect("never fails")
    }

    fn tell(&mut self, param: Self::Param, value: Self::Value) {
        let o = Observation { param, value };
        let i = self
            .observations
            .binary_search_by(|x| x.value.cmp(&o.value))
            .unwrap_or_else(|i| i);
        self.observations.insert(i, o);
    }
}
