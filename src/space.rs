use std::ops::Range;

pub trait SearchSpace {
    type ExternalParam;
    type InternalParam;

    fn internal_range(&self) -> Range<Self::InternalParam>;
    fn to_internal(&self, param: &Self::ExternalParam) -> Self::InternalParam;
    fn to_external(&self, param: &Self::InternalParam) -> Self::ExternalParam;
}

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
