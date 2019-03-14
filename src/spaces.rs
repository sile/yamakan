use crate::SearchSpace;
use std::ops::Range;

#[derive(Debug, Default)]
pub struct Bool;
impl SearchSpace for Bool {
    type ExternalParam = bool;
    type InternalParam = usize;

    fn internal_range(&self) -> Range<Self::InternalParam> {
        Range { start: 0, end: 2 }
    }

    fn to_internal(&self, param: &Self::ExternalParam) -> Self::InternalParam {
        *param as usize
    }

    fn to_external(&self, param: &Self::InternalParam) -> Self::ExternalParam {
        debug_assert!(*param < 2);
        *param != 0
    }
}

#[derive(Debug, Default)]
pub struct UniformF64 {
    pub low: f64,  // inclusive
    pub high: f64, // exclusive
}
impl SearchSpace for UniformF64 {
    type ExternalParam = f64;
    type InternalParam = f64;

    fn internal_range(&self) -> Range<Self::InternalParam> {
        Range {
            start: self.low,
            end: self.high,
        }
    }

    fn to_internal(&self, param: &Self::ExternalParam) -> Self::InternalParam {
        *param
    }

    fn to_external(&self, param: &Self::InternalParam) -> Self::ExternalParam {
        *param
    }
}
