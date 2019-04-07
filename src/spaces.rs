use crate::ParamSpace;
use rand::distributions::Distribution;
use rand::Rng;
use std::ops::Range;

pub trait PriorDistribution {
    type Param;
    type Distribution: Distribution<Self::Param>;

    fn prior(&self) -> &Self::Distribution;
}

#[derive(Debug, Default, Clone)]
pub struct Bool;
impl PriorDistribution for Bool {
    type Param = bool;
    type Distribution = Self;

    fn prior(&self) -> &Self::Distribution {
        self
    }
}
impl Distribution<bool> for Bool {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen::<bool>()
    }
}
impl ParamSpace for Bool {
    type External = bool;
    type Internal = usize;

    fn range(&self) -> Range<Self::Internal> {
        Range { start: 0, end: 2 }
    }

    fn internalize(&self, param: &Self::External) -> Self::Internal {
        *param as usize
    }

    fn externalize(&self, param: &Self::Internal) -> Self::External {
        debug_assert!(*param < 2);
        *param != 0
    }
}

#[derive(Debug, Default, Clone)]
pub struct F64 {
    pub low: f64,  // inclusive
    pub high: f64, // exclusive
}
impl ParamSpace for F64 {
    type External = f64;
    type Internal = f64;

    fn range(&self) -> Range<Self::Internal> {
        Range {
            start: self.low,
            end: self.high,
        }
    }

    fn internalize(&self, param: &Self::External) -> Self::Internal {
        *param
    }

    fn externalize(&self, param: &Self::Internal) -> Self::External {
        *param
    }
}
