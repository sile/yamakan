use std::ops::Range;

pub trait ParamSpace {
    type External;
    type Internal;
    // type PRIOR;

    fn internalize(&self, param: &Self::External) -> Self::Internal;
    fn externalize(&self, param: &Self::Internal) -> Self::External;
    fn range(&self) -> Range<Self::Internal>;
}
