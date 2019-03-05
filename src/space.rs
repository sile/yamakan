use std::num::NonZeroUsize;

pub trait NumericalSpace {
    type Param;
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
