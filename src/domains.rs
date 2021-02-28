//! Parameter search domains.
use crate::{Domain, ErrorKind, Result};
use ordered_float::NotNan;
use rand::distributions::Distribution;
use rand::Rng;
use std::num::NonZeroU64;

/// Vector domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecDomain<T>(pub Vec<T>);

impl<T: Domain> Domain for VecDomain<T> {
    type Point = Vec<T::Point>;
}

impl<T> Distribution<Vec<T::Point>> for VecDomain<T>
where
    T: Domain + Distribution<<T as Domain>::Point>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<T::Point> {
        self.0.iter().map(|t| t.sample(rng)).collect()
    }
}

/// Categorical domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CategoricalDomain {
    cardinality: NonZeroU64,
}
impl CategoricalDomain {
    /// Makes a new `CategoricalDomain` instance.
    ///
    /// # Errors
    ///
    /// If `cardinality` is `0`, this function returns an `ErrorKind::InvalidInput` error.
    pub fn new(cardinality: u64) -> Result<Self> {
        let cardinality = track_assert_some!(NonZeroU64::new(cardinality), ErrorKind::InvalidInput);
        Ok(Self { cardinality })
    }

    /// Returns the cardinality of this domain.
    pub const fn cardinality(&self) -> NonZeroU64 {
        self.cardinality
    }
}
impl Domain for CategoricalDomain {
    type Point = u64;
}
impl From<NonZeroU64> for CategoricalDomain {
    fn from(cardinality: NonZeroU64) -> Self {
        Self { cardinality }
    }
}
impl Distribution<u64> for CategoricalDomain {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        rng.gen_range(0..self.cardinality.get())
    }
}

/// Discrete numerical domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiscreteDomain {
    size: NonZeroU64,
}
impl DiscreteDomain {
    /// Makes a new `DiscreteDomain` instance.
    ///
    /// # Errors
    ///
    /// If `size` is `0`, this function returns an `ErrorKind::InvalidInput` error.
    pub fn new(size: u64) -> Result<Self> {
        let size = track_assert_some!(NonZeroU64::new(size), ErrorKind::InvalidInput);
        Ok(Self { size })
    }

    /// Returns the size of this domain.
    pub const fn size(&self) -> NonZeroU64 {
        self.size
    }
}
impl Domain for DiscreteDomain {
    type Point = u64;
}
impl From<NonZeroU64> for DiscreteDomain {
    fn from(size: NonZeroU64) -> Self {
        Self { size }
    }
}
impl Distribution<u64> for DiscreteDomain {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        rng.gen_range(0..self.size.get())
    }
}

/// Continuous numerical domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContinuousDomain {
    low: NotNan<f64>,
    high: NotNan<f64>,
}
impl ContinuousDomain {
    /// Makes a new `ContinuousDomain` instance.
    ///
    /// The returned instance represents a half-closed interval, i.e., `[low..high)`.
    ///
    /// # Errors
    ///
    /// If one of the following conditions is satisfied, this function returns an `ErrorKind::InvalidInput` error:
    ///
    /// - `low` or `high` is not a finite number
    /// - `low >= high`
    /// - `high - low` is not a finite number
    pub fn new(low: f64, high: f64) -> Result<Self> {
        track_assert!(low.is_finite(), ErrorKind::InvalidInput; low, high);
        track_assert!(high.is_finite(), ErrorKind::InvalidInput; low, high);
        track_assert!(low < high, ErrorKind::InvalidInput; low, high);
        track_assert!((high - low).is_finite(), ErrorKind::InvalidInput; low, high);

        Ok(unsafe {
            Self {
                low: NotNan::unchecked_new(low),
                high: NotNan::unchecked_new(high),
            }
        })
    }

    /// Returns the lower bound of this domain.
    pub fn low(&self) -> f64 {
        self.low.into_inner()
    }

    /// Returns the upper bound of this domain.
    pub fn high(&self) -> f64 {
        self.high.into_inner()
    }

    /// Returns the size of this domain.
    pub fn size(&self) -> f64 {
        self.high() - self.low()
    }
}
impl Domain for ContinuousDomain {
    type Point = f64;
}
impl Distribution<f64> for ContinuousDomain {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.gen_range(self.low()..self.high())
    }
}
