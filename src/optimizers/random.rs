use crate::optimizer::Optimizer;
use crate::space::ParamSpace;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct RandomOptimizer<S, V> {
    space: S,
    _value: PhantomData<V>,
}
impl<S, V> RandomOptimizer<S, V>
where
    S: ParamSpace,
    S::Internal: SampleUniform,
{
    pub fn new(space: S) -> Self {
        Self {
            space,
            _value: PhantomData,
        }
    }

    pub fn space(&self) -> &S {
        &self.space
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
        let r = self.space.internal_range();
        let i = rng.gen_range(r.start, r.end);
        self.space.externalize(&i)
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
