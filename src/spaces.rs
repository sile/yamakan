use crate::range::Range;
use rand::distributions::Distribution;
use rand::Rng;

pub trait ParamSpace {
    type Param;
}

/// This trait allows for sampling a parameter from the prior distribution of a parameter space.
pub trait PriorDistribution: ParamSpace + Distribution<<Self as ParamSpace>::Param> {}

pub trait PriorPmf: ParamSpace {
    fn pmf(&self, param: &Self::Param) -> f64;

    fn ln_pmf(&self, param: &Self::Param) -> f64 {
        self.pmf(param).ln()
    }
}

pub trait PriorPdf: ParamSpace {
    fn pdf(&self, param: &Self::Param) -> f64;

    fn ln_pdf(&self, param: &Self::Param) -> f64 {
        self.pdf(param).ln()
    }
}

pub trait PriorCdf: ParamSpace {
    fn cdf(&self, param: &Self::Param) -> f64;
}

pub trait Categorical: ParamSpace {
    fn size(&self) -> usize;

    fn into_index(&self, param: Self::Param) -> usize;

    fn from_index(&self, index: usize) -> Self::Param;
}

pub trait Numerical: ParamSpace {
    fn range(&self) -> Range<f64>;

    fn into_f64(&self, param: Self::Param) -> f64;

    fn from_f64(&self, n: f64) -> Self::Param;
}

#[derive(Debug, Default, Clone)]
pub struct Bool;
impl Distribution<bool> for Bool {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen::<bool>()
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

    fn into_index(&self, param: Self::Param) -> usize {
        param as usize
    }

    fn from_index(&self, index: usize) -> Self::Param {
        index != 0
    }
}

#[derive(Debug, Default, Clone)]
pub struct F64 {
    pub low: f64,  // inclusive
    pub high: f64, // exclusive
}
