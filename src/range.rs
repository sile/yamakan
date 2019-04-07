use crate::{ErrorKind, Result};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct Range<T> {
    pub low: T,  // inclusive
    pub high: T, // exclusive
}
impl<T> Range<T>
where
    T: PartialOrd,
{
    pub fn new(low: T, high: T) -> Result<Self> {
        track_assert_eq!(
            low.partial_cmp(&high),
            Some(Ordering::Less),
            ErrorKind::InvalidInput
        );
        Ok(Self { low, high })
    }

    pub fn contains(&self, x: &T) -> bool {
        match (self.low.partial_cmp(x), self.high.partial_cmp(x)) {
            (Some(Ordering::Equal), Some(Ordering::Greater))
            | (Some(Ordering::Less), Some(Ordering::Greater)) => true,
            _ => false,
        }
    }
}
