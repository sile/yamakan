use crate::optimizer::Optimizer;
use crate::space::{CategoricalSpace, NumericalSpace};
use rand::Rng;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct RandomNumericalOptimizer<P, V> {
    param_space: P,
    _value: PhantomData<V>,
}
impl<P, V> Optimizer for RandomNumericalOptimizer<P, V>
where
    P: NumericalSpace,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let r = self.param_space.internal_range();
        let i = rng.gen_range(r.start, r.end);
        self.param_space.internal_to_param(i)
    }

    fn tell(&mut self, _param: Self::Param, _value: Self::Value) {}
}

#[derive(Debug)]
pub struct RandomCategoricalOptimizer<P, V> {
    param_space: P,
    _value: PhantomData<V>,
}
impl<P, V> Optimizer for RandomCategoricalOptimizer<P, V>
where
    P: CategoricalSpace,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let i = rng.gen_range(0, self.param_space.size().get());
        self.param_space.index_to_param(i)
    }

    fn tell(&mut self, _param: Self::Param, _value: Self::Value) {}
}
