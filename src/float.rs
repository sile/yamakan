use std::cmp::{self, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NonNanF64(f64);
impl NonNanF64 {
    pub fn new(x: f64) -> Self {
        assert!(!x.is_nan());
        Self(x)
    }

    pub fn as_f64(self) -> f64 {
        self.0
    }
}
impl Eq for NonNanF64 {}
impl Ord for NonNanF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or_else(|| unreachable!())
    }
}

pub fn max(x: f64, y: f64) -> f64 {
    cmp::max(NonNanF64::new(x), NonNanF64::new(y)).as_f64()
}

pub fn min(x: f64, y: f64) -> f64 {
    cmp::min(NonNanF64::new(x), NonNanF64::new(y)).as_f64()
}

pub fn clip(min_x: f64, x: f64, max_x: f64) -> f64 {
    max(min(max_x, x), min_x)
}
