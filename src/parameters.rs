//! Parameter spaces.
use crate::{ErrorKind, Result};
use rand::distributions::Distribution;
use rand::Rng;
use rustats::num::FiniteF64;
use rustats::range::Range;

/// This trait allows for defining a parameter space.
pub trait ParamSpace {
    /// Concrete parameter type.
    type Param;
}

/// This trait allows for sampling a parameter from the prior distribution of a parameter space.
pub trait PriorDistribution: ParamSpace + Distribution<<Self as ParamSpace>::Param> {}

/// This trait allows for defining a categorical parameter space.
pub trait Categorical: ParamSpace {
    /// Returns the number of possible categories.
    fn size(&self) -> usize;

    /// Converts the given parameter (category) to the associated index.
    fn encode(&self, param: &Self::Param) -> Result<usize>;

    /// Converts the given index to the associated parameter (category).
    fn decode(&self, index: usize) -> Result<Self::Param>;
}

/// This trait allows for defining a discrete numerical parameter space.
pub trait Discrete: ParamSpace {
    /// Returns the number of possible discrete entries.
    fn size(&self) -> u64;

    /// Converts the given parameter to the associated index.
    fn encode(&self, param: &Self::Param) -> Result<u64>;

    /// Converts the given index to the associated parameter.
    fn decode(&self, index: u64) -> Result<Self::Param>;
}

/// This trait allows for defining a continuous numerical parameter space.
pub trait Continuous: ParamSpace {
    /// Returns the range of the internal representation of this parameter space.
    fn range(&self) -> Range<FiniteF64>;

    /// Converts the given parameter to the associated internal value.
    fn encode(&self, param: &Self::Param) -> Result<FiniteF64>;

    /// Converts the given internal value to the associated parameter.
    fn decode(&self, value: FiniteF64) -> Result<Self::Param>;
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

    fn encode(&self, param: &Self::Param) -> Result<usize> {
        Ok(*param as usize)
    }

    fn decode(&self, index: usize) -> Result<Self::Param> {
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

/// 64-bit unsigned integer number parameter space.
#[derive(Debug, Clone, Copy)]
pub struct U64(u64);
impl U64 {
    /// Makes a `U64` parameter space.
    pub const fn new(size: u64) -> Self {
        Self(size)
    }
}
impl ParamSpace for U64 {
    type Param = u64;
}
impl Discrete for U64 {
    fn size(&self) -> u64 {
        self.0
    }

    fn encode(&self, param: &Self::Param) -> Result<u64> {
        Ok(*param)
    }

    fn decode(&self, index: u64) -> Result<Self::Param> {
        Ok(index)
    }
}
impl Distribution<u64> for U64 {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        rng.gen_range(0, self.0)
    }
}
impl PriorDistribution for U64 {}

/// 64-bit floating point number parameter space.
#[derive(Debug, Clone, Copy)]
pub struct F64(Range<FiniteF64>);
impl F64 {
    /// Makes an `F64` parameter space with the given range.
    ///
    /// # Errors
    ///
    /// If all of the following conditions are not satisfied, an `ErrorKind::InvalidInput` error will be returned:
    /// - `low` and `high` are finite numbers
    /// - `low` is smaller than `high`
    pub fn new(low: f64, high: f64) -> Result<Self> {
        let low = track!(FiniteF64::new(low))?;
        let high = track!(FiniteF64::new(high))?;
        let range = track!(Range::new(low, high); low, high)?;
        Ok(Self(range))
    }
}
impl From<Range<FiniteF64>> for F64 {
    fn from(f: Range<FiniteF64>) -> Self {
        Self(f)
    }
}
impl ParamSpace for F64 {
    type Param = f64;
}
impl Continuous for F64 {
    fn range(&self) -> Range<FiniteF64> {
        self.0
    }

    fn encode(&self, param: &Self::Param) -> Result<FiniteF64> {
        let param = track!(FiniteF64::new(*param))?;
        track_assert!(self.0.contains(&param), ErrorKind::InvalidInput; param);
        Ok(param)
    }

    fn decode(&self, value: FiniteF64) -> Result<Self::Param> {
        track_assert!(self.0.contains(&value), ErrorKind::InvalidInput; value);
        Ok(value.get())
    }
}
impl Distribution<f64> for F64 {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.gen_range(self.0.low.get(), self.0.high.get())
    }
}
impl PriorDistribution for F64 {}
