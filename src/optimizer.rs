use crate::observation::{IdGen, Obs, ObsId};
use crate::Result;
use rand::Rng;

pub trait Optimizer {
    type Param;
    type Value;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param>>;

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()>;

    fn forget(&mut self, id: ObsId) -> Result<()>;
}
