use rand::Rng;

pub trait Optimizer {
    type Param;
    type Value;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param;
    fn tell(&mut self, param: Self::Param, value: Self::Value);
}

#[derive(Debug)]
pub struct Observation<P, V> {
    pub param: P,
    pub value: V,
}
