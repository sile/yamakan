use crate::Result;
use rand::Rng;

pub trait Optimizer {
    type Param;
    type Value;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Result<Self::Param>;
    fn tell(&mut self, param: Self::Param, value: Self::Value) -> Result<()>;
}

#[derive(Debug)]
pub struct Observation<P, V> {
    pub param: P,
    pub value: V,
}
