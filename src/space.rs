use std::ops::Range;

// TODO: s/SearchSpace/ParamSpace/
pub trait SearchSpace {
    type ExternalParam;
    type InternalParam;

    fn internal_range(&self) -> Range<Self::InternalParam>;
    fn to_internal(&self, param: &Self::ExternalParam) -> Self::InternalParam;
    fn to_external(&self, param: &Self::InternalParam) -> Self::ExternalParam;
}
