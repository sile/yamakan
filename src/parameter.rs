// TODO: remove(?)
use std::num::NonZeroUsize;

pub trait Categorical {
    const SIZE: NonZeroUsize;
    fn to_index(&self) -> usize;
    fn from_index(index: usize) -> Self;
}
impl Categorical for bool {
    const SIZE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(2) };

    fn to_index(&self) -> usize {
        *self as usize
    }

    fn from_index(index: usize) -> Self {
        index != 0
    }
}
