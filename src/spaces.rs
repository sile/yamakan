use crate::range::Range;
use crate::{ErrorKind, Result};
use rand::distributions::Distribution;
use rand::Rng;

pub trait ParamSpace {
    type Param;
}

/// This trait allows for sampling a parameter from the prior distribution of a parameter space.
pub trait PriorDistribution: ParamSpace + Distribution<<Self as ParamSpace>::Param> {}

pub trait PriorPmf: ParamSpace {
    // TODO: use internal value?
    fn pmf(&self, param: &Self::Param) -> f64;

    fn ln_pmf(&self, param: &Self::Param) -> f64 {
        self.pmf(param).ln()
    }
}

pub trait PriorPdf: Numerical {
    fn pdf(&self, internal: f64) -> f64;

    fn ln_pdf(&self, internal: f64) -> f64 {
        self.pdf(internal).ln()
    }
}

pub trait PriorCdf: Numerical {
    fn cdf(&self, internal: f64) -> f64;
}

pub trait Categorical: ParamSpace {
    fn size(&self) -> usize;

    fn to_index(&self, param: &Self::Param) -> Result<usize>;

    fn from_index(&self, index: usize) -> Result<Self::Param>;
}

pub trait Numerical: ParamSpace {
    fn range(&self) -> Range<f64>;

    fn to_f64(&self, param: &Self::Param) -> Result<f64>;

    fn from_f64(&self, n: f64) -> Result<Self::Param>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Bool;
impl Distribution<bool> for Bool {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen()
    }
}
impl ParamSpace for Bool {
    type Param = bool;
}
impl PriorDistribution for Bool {}
impl PriorPmf for Bool {
    fn pmf(&self, _param: &Self::Param) -> f64 {
        0.5
    }
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

#[derive(Debug, Clone, Copy)]
pub struct F64(Range<f64>);
impl F64 {
    pub fn new(low: f64, high: f64) -> Result<Self> {
        track!(Range::new(low, high).map(Self); low, high)
    }
}
impl ParamSpace for F64 {
    type Param = f64;
}
impl Numerical for F64 {
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
impl PriorPdf for F64 {
    fn pdf(&self, _internal: f64) -> f64 {
        1.0 / self.0.width()
    }

    fn ln_pdf(&self, _internal: f64) -> f64 {
        1f64.ln() - self.0.width().ln()
    }
}
impl PriorCdf for F64 {
    fn cdf(&self, internal: f64) -> f64 {
        if internal < self.0.low {
            0.0
        } else if internal >= self.0.high {
            1.0
        } else {
            (internal - self.0.low) / self.0.width()
        }
    }
}
