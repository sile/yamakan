use crate::optimizer::Optimizer;
use crate::space::SearchSpace;
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
    S: SearchSpace,
    S::InternalParam: SampleUniform,
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
    S: SearchSpace,
    S::InternalParam: SampleUniform,
{
    type Param = S::ExternalParam;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let r = self.space.internal_range();
        let i = rng.gen_range(r.start, r.end);
        self.space.to_external(&i)
    }

    fn tell(&mut self, _param: Self::Param, _value: Self::Value) {}
}
impl<S, V> Default for RandomOptimizer<S, V>
where
    S: Default + SearchSpace,
    S::InternalParam: SampleUniform,
{
    fn default() -> Self {
        Self::new(S::default())
    }
}
