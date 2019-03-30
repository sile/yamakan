use crate::observation::{IdGenerator, Observation};
use crate::Result;
use rand::Rng;

pub trait Optimizer {
    type Param;
    type Value;

    fn ask<R: Rng, G: IdGenerator>(
        &mut self,
        rng: &mut R,
        idgen: &mut G,
    ) -> Result<Observation<Self::Param, ()>>;

    fn tell(&mut self, observation: Observation<Self::Param, Self::Value>) -> Result<()>;
}
