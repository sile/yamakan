use std::num::NonZeroUsize;
use std::ops::Range;

pub trait NumericalSpace {
    type Param;

    // TODO: q
    fn internal_range(&self) -> Range<f64>;
    fn param_to_internal(&self, param: &Self::Param) -> f64;
    fn internal_to_param(&self, internal_value: f64) -> Self::Param;
}

#[derive(Debug)]
pub struct UniformF64Space {
    pub low: f64,
    pub high: f64,
}
impl NumericalSpace for UniformF64Space {
    type Param = f64;

    fn internal_range(&self) -> Range<f64> {
        Range {
            start: self.low,
            end: self.high,
        }
    }

    fn param_to_internal(&self, param: &Self::Param) -> f64 {
        *param
    }

    fn internal_to_param(&self, internal_value: f64) -> Self::Param {
        internal_value
    }
}

pub trait CategoricalSpace {
    type Param;

    fn size(&self) -> NonZeroUsize;
    fn param_to_index(&self, param: &Self::Param) -> usize;
    fn index_to_param(&self, index: usize) -> Self::Param;
}

#[derive(Debug)]
pub struct BoolSpace;
impl CategoricalSpace for BoolSpace {
    type Param = bool;

    fn size(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(2) }
    }

    fn param_to_index(&self, param: &Self::Param) -> usize {
        *param as usize
    }

    fn index_to_param(&self, index: usize) -> Self::Param {
        debug_assert!(index < 2);
        index != 0
    }
}
