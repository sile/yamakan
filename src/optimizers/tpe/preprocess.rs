use crate::float;
use crate::iter::linspace;
use crate::observation::Obs;
use crate::range::Range;
use std::cmp;
use std::iter::{once, repeat};

// TODO: rename
pub trait Preprocess<V> {
    fn divide_observations<P>(&self, observations: &[&Obs<P, V>]) -> usize;
    fn weight_observations<P>(
        &self,
        observations: &[&Obs<P, V>],
        is_superior: bool,
    ) -> Box<dyn Iterator<Item = f64>>;

    fn sigmas<'a, I>(&self, range: Range<f64>, mus: I) -> Box<dyn Iterator<Item = f64> + 'a>
    where
        I: ExactSizeIterator<Item = f64> + 'a;
}

#[derive(Debug)]
pub struct DefaultPreprocessor {
    pub divide_factor: f64,
    pub uniform_sigma: bool,
}
impl Default for DefaultPreprocessor {
    fn default() -> Self {
        Self {
            divide_factor: 0.25,
            uniform_sigma: false,
        }
    }
}
impl<V> Preprocess<V> for DefaultPreprocessor {
    fn divide_observations<P>(&self, observations: &[&Obs<P, V>]) -> usize {
        let n = observations.len() as f64;
        cmp::min((self.divide_factor * n.sqrt()).ceil() as usize, 25)
    }

    fn weight_observations<P>(
        &self,
        observations: &[&Obs<P, V>],
        is_superior: bool,
    ) -> Box<dyn Iterator<Item = f64>> {
        let n = observations.len();
        if is_superior {
            Box::new(repeat(1.0).take(n))
        } else {
            let m = cmp::max(n, 25) - 25;
            Box::new(linspace(1.0 / (n as f64), 1.0, m).chain(repeat(1.0).take(n - m)))
        }
    }

    fn sigmas<'a, I>(&self, range: Range<f64>, mus: I) -> Box<dyn Iterator<Item = f64> + 'a>
    where
        I: ExactSizeIterator<Item = f64> + 'a,
    {
        assert_ne!(mus.len(), 0); // TODO

        if self.uniform_sigma {
            let sigma = range.width() / (mus.len()) as f64;
            return Box::new(repeat(sigma).take(mus.len()));
        }

        let maxsigma = range.width();
        let minsigma = range.width() / float::min(100.0, 1.0 + (mus.len() as f64));

        let sigmas = Windows::new(once(range.low).chain(mus).chain(once(range.high)))
            .map(move |x| float::clip(minsigma, float::max(x.1 - x.0, x.2 - x.1), maxsigma));
        Box::new(sigmas)
    }
}

struct Windows<I> {
    inner: I,
    prev: f64,
    curr: f64,
}
impl<I: Iterator<Item = f64>> Windows<I> {
    fn new(mut inner: I) -> Self {
        let prev = inner.next().unwrap_or_else(|| unreachable!());
        let curr = inner.next().unwrap_or_else(|| unreachable!());
        Self { inner, prev, curr }
    }
}
impl<I: Iterator<Item = f64>> Iterator for Windows<I> {
    type Item = (f64, f64, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.inner.next() {
            let prev = self.prev;
            let curr = self.curr;
            self.prev = curr;
            self.curr = next;
            Some((prev, curr, next))
        } else {
            None
        }
    }
}
