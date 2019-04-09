use super::parzen_estimator::Sample;
use crate::iter::linspace;
use crate::observation::Obs;
use std::cmp;
use std::f64::EPSILON;
use std::iter::repeat;
use std::num::NonZeroUsize;

pub trait KdeStrategy {
    fn kde_bandwidth(&self, samples: &[Sample]) -> f64;
}

pub trait Strategy<P, V> {
    fn division_position(&self, obss: &[&Obs<P, V>]) -> usize;

    fn prior_weight(&self, obss: &[&Obs<P, V>]) -> f64;

    fn superior_weights(&self, obss: &[&Obs<P, V>]) -> Box<dyn Iterator<Item = f64>>;

    fn inferior_weights(&self, obss: &[&Obs<P, V>]) -> Box<dyn Iterator<Item = f64>>;
}

pub trait CategoricalStrategy<V>: Strategy<usize, V> {}

pub trait NumericalStrategy<V>: Strategy<f64, V> + KdeStrategy {
    fn ei_candidates(&self, obss: &[&Obs<f64, V>]) -> NonZeroUsize;
}

// TODO: rename: s/default/built-in/ (?)
#[derive(Debug)]
pub struct DefaultStrategy {
    divide_factor: f64,
    max_superiors: NonZeroUsize,
    prior_weight: f64,
    ei_candidates: NonZeroUsize,
}
impl Default for DefaultStrategy {
    fn default() -> Self {
        Self {
            divide_factor: 1.0,
            max_superiors: unsafe { NonZeroUsize::new_unchecked(25) },
            prior_weight: 1.0,
            ei_candidates: unsafe { NonZeroUsize::new_unchecked(4) },
        }
    }
}
impl<P, V> Strategy<P, V> for DefaultStrategy {
    fn division_position(&self, obss: &[&Obs<P, V>]) -> usize {
        let n = obss.len() as f64;
        cmp::min(
            (self.divide_factor * n.sqrt()).ceil() as usize,
            self.max_superiors.get(),
        )
    }

    fn prior_weight(&self, _obss: &[&Obs<P, V>]) -> f64 {
        self.prior_weight
    }

    fn superior_weights(&self, obss: &[&Obs<P, V>]) -> Box<dyn Iterator<Item = f64>> {
        Box::new(repeat(1.0).take(obss.len()))
    }

    fn inferior_weights(&self, obss: &[&Obs<P, V>]) -> Box<dyn Iterator<Item = f64>> {
        let n = obss.len();
        let m = cmp::max(n, 25) - 25; // TODO: change
        Box::new(linspace(1.0 / (n as f64), 1.0, m).chain(repeat(1.0).take(n - m)))
    }
}
impl<V> CategoricalStrategy<V> for DefaultStrategy {}
impl KdeStrategy for DefaultStrategy {
    fn kde_bandwidth(&self, samples: &[Sample]) -> f64 {
        // TODO:

        // Silverman’s rule of thumb
        let n = samples.len() as f64;
        let mut sd = sd(samples.iter().map(|o| o.mu));
        if sd == 0.0 {
            sd = EPSILON;
        }
        1.06 * sd * n.powf(-1.0 / 5.0)
    }
}
impl<V> NumericalStrategy<V> for DefaultStrategy {
    fn ei_candidates(&self, _obss: &[&Obs<f64, V>]) -> NonZeroUsize {
        self.ei_candidates
    }
}

// TODO: move
fn sd<I>(xs: I) -> f64
where
    I: ExactSizeIterator<Item = f64> + Clone,
{
    let n = xs.len() as f64;
    let sum = xs.clone().into_iter().sum::<f64>();
    let avg = sum / n;
    let var = xs.into_iter().map(|x| (x - avg).powi(2)).sum::<f64>() / n;
    var.sqrt()
}
