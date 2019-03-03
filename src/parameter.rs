pub trait Categorical {
    const MAX_INDEX: usize;
    fn to_index(&self) -> usize;
    fn from_index(index: usize) -> Self;
}
impl Categorical for bool {
    const MAX_INDEX: usize = 1;

    fn to_index(&self) -> usize {
        *self as usize
    }

    fn from_index(index: usize) -> Self {
        index != 0
    }
}
