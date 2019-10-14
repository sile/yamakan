//! Random optimizer.
use crate::{Domain, IdGen, Obs, Optimizer, Result};
use rand::distributions::Distribution;
use rand::Rng;
use std::marker::PhantomData;

/// Random optimizer.
///
/// This optimizer samples parameters at random from the given domain.
#[derive(Debug)]
pub struct RandomOptimizer<P, V> {
    param_domain: P,
    _value: PhantomData<V>,
}
impl<P, V> RandomOptimizer<P, V>
where
    P: Domain + Distribution<<P as Domain>::Point>,
{
    /// Makes a new `RandomOptimizer` instance.
    pub fn new(param_domain: P) -> Self {
        Self {
            param_domain,
            _value: PhantomData,
        }
    }
}
impl<P, V> Optimizer for RandomOptimizer<P, V>
where
    P: Domain + Distribution<<P as Domain>::Point>,
{
    type Param = P::Point;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, mut rng: R, idg: G) -> Result<Obs<Self::Param>> {
        track!(Obs::new(idg, self.param_domain.sample(&mut rng)))
    }

    fn tell(&mut self, _obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        Ok(())
    }
}
impl<P, V> Default for RandomOptimizer<P, V>
where
    P: Default + Domain + Distribution<<P as Domain>::Point>,
{
    fn default() -> Self {
        Self::new(P::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::DiscreteDomain;
    use crate::generators::SerialIdGenerator;
    use rand;
    use trackable::result::TestResult;

    #[test]
    fn random_works() -> TestResult {
        let mut opt = RandomOptimizer::new(track!(DiscreteDomain::new(10))?);
        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.tell(obs))?;

        Ok(())
    }
}
