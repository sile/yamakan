use super::{Observation, TpeStrategy};
use crate::optimizer::Optimizer;
use crate::space::NumericalSpace;
use rand::Rng;

#[derive(Debug)]
pub struct TpeNumericalOptimizer<P, V, S>
where
    P: NumericalSpace,
{
    param_space: P,
    strategy: S,
    observations: Vec<Observation<P::Param, V>>,
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

        panic!()
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
