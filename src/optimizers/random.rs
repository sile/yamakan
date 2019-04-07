//! Random optimizer.
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::space::ParamSpace;
use crate::Result;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::marker::PhantomData;

/// Random optimizer.
#[derive(Debug)]
pub struct RandomOptimizer<P, V> {
    param_space: P,
    _value: PhantomData<V>,
}
impl<P, V> RandomOptimizer<P, V>
where
    P: ParamSpace,
    P::Internal: SampleUniform,
{
    pub fn new(param_space: P) -> Self {
        Self {
            param_space,
            _value: PhantomData,
        }
    }

    pub fn param_space(&self) -> &P {
        &self.param_space
    }
}
impl<P, V> Optimizer for RandomOptimizer<P, V>
where
    P: ParamSpace,
    P::Internal: SampleUniform,
{
    type Param = P::External;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        let r = self.param_space.range();
        let i = rng.gen_range(r.start, r.end);
        track!(Obs::new(idg, self.param_space.externalize(&i)))
    }

    fn tell(&mut self, _obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        Ok(())
    }

    fn forget(&mut self, _id: ObsId) -> Result<()> {
        Ok(())
    }
}
impl<P, V> Default for RandomOptimizer<P, V>
where
    P: Default + ParamSpace,
    P::Internal: SampleUniform,
{
    fn default() -> Self {
        Self::new(P::default())
    }
}
