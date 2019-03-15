use std::ops::Range;

pub trait ParamSpace {
    type External;
    type Internal;

    fn internal_range(&self) -> Range<Self::Internal>;
    fn internalize(&self, param: &Self::External) -> Self::Internal;
    fn externalize(&self, param: &Self::Internal) -> Self::External;
}
