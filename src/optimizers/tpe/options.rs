use crate::{Error, ErrorKind};
use std::num::NonZeroUsize;

#[derive(Debug)]
pub struct TpeOptions<T> {
    pub(crate) preprocessor: T,
    pub(crate) prior_weight: f64,
    pub(crate) ei_candidates: NonZeroUsize,
    pub(crate) prior_uniform: bool,
    pub(crate) uniform_sigma: bool,
}
impl<T> TpeOptions<T> {
    pub fn new(preprocessor: T) -> Self {
        Self {
            preprocessor,
            prior_weight: 1.0,
            ei_candidates: unsafe { NonZeroUsize::new_unchecked(24) },
            prior_uniform: false,
            uniform_sigma: false,
        }
    }

    pub fn prior_weight(mut self, weight: f64) -> Result<Self, Error> {
        track_assert!(weight > 0.0, ErrorKind::InvalidInput; weight);
        track_assert!(weight.is_finite(), ErrorKind::InvalidInput; weight);

        self.prior_weight = weight;
        Ok(self)
    }

    pub fn ei_candidates(mut self, n: NonZeroUsize) -> Self {
        self.ei_candidates = n;
        self
    }

    pub fn prior_uniform(mut self, b: bool) -> Self {
        self.prior_uniform = b;
        self
    }

    pub fn uniform_sigma(mut self, b: bool) -> Self {
        self.uniform_sigma = b;
        self
    }
}
impl<T: Default> Default for TpeOptions<T> {
    fn default() -> Self {
        TpeOptions {
            preprocessor: T::default(),
            prior_weight: 1.0,
            ei_candidates: unsafe { NonZeroUsize::new_unchecked(24) },
            prior_uniform: false,
            uniform_sigma: false,
        }
    }
}
