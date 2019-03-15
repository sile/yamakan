use crate::optimizer::Optimizer;
use crate::space::ParamSpace;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct RandomOptimizer<S, V> {
    param_space: S,
    _value: PhantomData<V>,
}
impl<S, V> RandomOptimizer<S, V>
where
    S: ParamSpace,
    S::Internal: SampleUniform,
{
    pub fn new(param_space: S) -> Self {
        Self {
            param_space,
            _value: PhantomData,
        }
    }

    pub fn param_space(&self) -> &S {
        &self.param_space
    }
}
impl<S, V> Optimizer for RandomOptimizer<S, V>
where
    S: ParamSpace,
    S::Internal: SampleUniform,
{
    type Param = S::External;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let r = self.param_space.internal_range();
        let i = rng.gen_range(r.start, r.end);
        self.param_space.externalize(&i)
    }

    fn tell(&mut self, _param: Self::Param, _value: Self::Value) {}
}
impl<S, V> Default for RandomOptimizer<S, V>
where
    S: Default + ParamSpace,
    S::Internal: SampleUniform,
{
    fn default() -> Self {
        Self::new(S::default())
    }
}
