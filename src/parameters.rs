//! Parameter spaces.
use crate::{ErrorKind, Result};
use rand::distributions::Distribution;
use rand::Rng;
use rustats::range::Range;

/// Parameter space definition.
pub trait ParamSpace {
    /// Concrete parameter type.
    type Param;
}

/// This trait allows for sampling a parameter from the prior distribution of a parameter space.
pub trait PriorDistribution: ParamSpace + Distribution<<Self as ParamSpace>::Param> {}

pub trait Categorical: ParamSpace {
    fn size(&self) -> usize;

    fn to_index(&self, param: &Self::Param) -> Result<usize>;

    fn from_index(&self, index: usize) -> Result<Self::Param>;
}

pub trait Discrete: ParamSpace {
    fn range(&self) -> Range<u64>;

    // fn to_f64(&self, param: &Self::Param) -> Result<f64>;

    // fn from_f64(&self, n: f64) -> Result<Self::Param>;
}

pub trait Continuous: ParamSpace {
    fn range(&self) -> Range<f64>;

    fn to_f64(&self, param: &Self::Param) -> Result<f64>;

    fn from_f64(&self, n: f64) -> Result<Self::Param>;
}

/// Boolean parameter space.
#[derive(Debug, Default, Clone, Copy)]
pub struct Bool;
impl ParamSpace for Bool {
    type Param = bool;
}
impl Categorical for Bool {
    fn size(&self) -> usize {
        2
    }

    fn to_index(&self, param: &Self::Param) -> Result<usize> {
        Ok(*param as usize)
    }

    fn from_index(&self, index: usize) -> Result<Self::Param> {
        match index {
            0 => Ok(false),
            1 => Ok(true),
            _ => track_panic!(ErrorKind::InvalidInput; index),
        }
    }
}
impl Distribution<bool> for Bool {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen()
    }
}
impl PriorDistribution for Bool {}

/// 64-bit floating point number parameter space.
#[derive(Debug, Clone, Copy)]
pub struct F64(Range<f64>);
impl F64 {
    pub fn new(low: f64, high: f64) -> Result<Self> {
        let r = track!(Range::new(low, high); low, high)?;
        Ok(Self(r))
    }
}
impl ParamSpace for F64 {
    type Param = f64;
}
impl Continuous for F64 {
    fn range(&self) -> Range<f64> {
        self.0
    }

    fn to_f64(&self, param: &Self::Param) -> Result<f64> {
        track_assert!(self.0.contains(param), ErrorKind::InvalidInput; param);
        Ok(*param)
    }

    fn from_f64(&self, n: f64) -> Result<Self::Param> {
        track_assert!(self.0.contains(&n), ErrorKind::InvalidInput; n);
        Ok(n)
    }
}
impl Distribution<f64> for F64 {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.gen_range(self.0.low, self.0.high)
    }
}
impl PriorDistribution for F64 {}
