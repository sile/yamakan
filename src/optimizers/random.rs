//! Random optimizer.
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::spaces::PriorDistribution;
use crate::Result;
use rand::Rng;
use std::marker::PhantomData;

/// Random optimizer.
///
/// This optimizer samples parameters at random from the given prior distribution.
// TODO: remove `V = ()`
#[derive(Debug)]
pub struct RandomOptimizer<P, V = ()> {
    param_space: P,
    _value: PhantomData<V>,
}
impl<P, V> RandomOptimizer<P, V>
where
    P: PriorDistribution,
{
    /// Makes a new `RandomOptimizer` instance.
    pub fn new(param_space: P) -> Self {
        Self {
            param_space,
            _value: PhantomData,
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
impl<P, V> Optimizer for RandomOptimizer<P, V>
where
    P: PriorDistribution,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        track!(Obs::new(idg, self.param_space.sample(rng)))
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
    P: PriorDistribution + Default,
{
    fn default() -> Self {
        Self::new(P::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::SerialIdGenerator;
    use crate::spaces::Bool;
    use rand;
    use trackable::result::TestResult;

    #[test]
    fn random_works() -> TestResult {
        let mut opt = RandomOptimizer::new(Bool);
        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.tell(obs))?;

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.forget(obs.id))?;

        Ok(())
    }
}
