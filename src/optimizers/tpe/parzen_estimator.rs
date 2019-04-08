use super::Preprocess;
use crate::float::NonNanF64;
use crate::range::Range;
use crate::spaces::{Numerical, PriorCdf, PriorDistribution, PriorPdf};
use rand;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::distribution::{Continuous, Normal, Univariate};
use std::cmp;

#[derive(Debug)]
pub struct ParzenEstimatorBuilder {
    prior_weight: f64,
}
impl ParzenEstimatorBuilder {
    pub fn new(prior_weight: f64) -> Self {
        Self { prior_weight }
    }

    pub fn finish<'a, P, M, W, T, V>(
        &self,
        param_space: &'a P,
        mus: M,
        weights: W,
        range: Range<f64>,
        pp: &T,
    ) -> ParzenEstimator<'a, P>
    where
        M: Iterator<Item = f64>,
        W: Iterator<Item = f64>,
        P: Numerical + PriorCdf + PriorPdf,
        T: Preprocess<V>,
    {
        let mut entries = self.make_sorted_entries(mus, weights);
        self.insert_prior_entry(&mut entries, param_space, range);

        self.normalize_weights(&mut entries);
        self.setup_sigmas(pp, &mut entries, range);

        let p_accept = entries
            .iter()
            .map(|e| (e.cdf(range.high) - e.cdf(range.low)) * e.weight())
            .sum::<f64>();

        ParzenEstimator {
            entries,
            range,
            p_accept,
        }
    }

    fn make_sorted_entries<'a, P, M, W>(&self, mus: M, weights: W) -> Vec<Entry<'a, P>>
    where
        M: Iterator<Item = f64>,
        W: Iterator<Item = f64>,
    {
        let mut entries = mus
            .zip(weights)
            .map(|(mu, weight)| Entry::new(mu, weight, 0.0))
            .collect::<Vec<_>>();
        entries.sort_by_key(|x| NonNanF64::new(x.mu()));
        entries
    }

    fn insert_prior_entry<'a, P>(
        &self,
        entries: &mut Vec<Entry<'a, P>>,
        param_space: &'a P,
        range: Range<f64>,
    ) {
        let prior_mu = range.middle();
        let pos = entries
            .binary_search_by(|x| {
                NonNanF64::new(x.mu())
                    .cmp(&NonNanF64::new(prior_mu))
                    .then(cmp::Ordering::Greater)
            })
            .unwrap_or_else(|i| i);
        let entry = Entry::Prior {
            param_space,
            mu: prior_mu,
            weight: self.prior_weight,
        };
        entries.insert(pos, entry);
    }

    fn normalize_weights<'a, P>(&self, entries: &mut [Entry<'a, P>]) {
        let weight_sum = entries.iter().map(|x| x.weight()).sum::<f64>();
        for x in entries {
            let w = x.weight() / weight_sum;
            x.set_weight(w);
        }
    }

    fn setup_sigmas<'a, P, T, V>(&self, pp: &T, entries: &mut [Entry<'a, P>], range: Range<f64>)
    where
        T: Preprocess<V>,
    {
        let sigmas = pp.sigmas(range, entries.iter().map(|x| x.mu()));
        let entries: &mut [Entry<'a, P>] = unsafe { &mut *(entries as *const _ as *mut _) };
        for (s, e) in sigmas.zip(entries.iter_mut()) {
            e.set_sigma(s);
        }
    }
}
impl Default for ParzenEstimatorBuilder {
    fn default() -> Self {
        Self { prior_weight: 1.0 }
    }
}

#[derive(Debug)]
pub struct ParzenEstimator<'a, P> {
    entries: Vec<Entry<'a, P>>,
    range: Range<f64>,
    p_accept: f64,
}
impl<'a, P> ParzenEstimator<'a, P> {
    pub fn gmm(&self) -> Gmm<P> {
        Gmm { estimator: self }
    }
}

fn logsumexp(xs: &[f64]) -> f64 {
    let max_x = xs
        .iter()
        .max_by_key(|&&x| NonNanF64::new(x))
        .expect("never fails");
    xs.iter().map(|&x| (x - max_x).exp()).sum::<f64>().ln() + max_x
}

/// Gaussian Mixture Model.
#[derive(Debug)]
pub struct Gmm<'a, P> {
    estimator: &'a ParzenEstimator<'a, P>,
}
impl<'a, P> Gmm<'a, P>
where
    P: Numerical + PriorPdf + PriorCdf,
{
    pub fn log_pdf(&self, param: f64) -> f64 {
        let mut xs = Vec::with_capacity(self.estimator.entries.len());
        for e in &self.estimator.entries {
            let log_density = e.log_pdf(param);
            let x = log_density + (e.weight() / self.estimator.p_accept).ln();
            xs.push(x);
        }
        logsumexp(&xs)
    }
}
impl<'a, P> Distribution<f64> for Gmm<'a, P>
where
    P: Numerical + PriorDistribution,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let entry = self
                .estimator
                .entries
                .choose_weighted(rng, |x| x.weight())
                .expect("never fails");
            if let Some(x) = entry
                .sample(rng)
                .filter(|x| self.estimator.range.contains(x))
            {
                return x;
            }
        }
    }
}

#[derive(Debug)]
enum Entry<'a, P> {
    Prior {
        param_space: &'a P,
        mu: f64, // TODO: delete
        weight: f64,
    },
    Sample {
        mu: f64,
        weight: f64,
        sigma: f64, // std-dev: TODO: delete
    },
}
impl<'a, P> Entry<'a, P>
where
    P: Numerical + PriorDistribution,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<f64> {
        match *self {
            Entry::Sample { mu, sigma, .. } => {
                Some(rand::distributions::Normal::new(mu, sigma).sample(rng))
            }
            Entry::Prior { param_space, .. } => param_space.to_f64(&param_space.sample(rng)).ok(),
        }
    }
}
impl<'a, P> Entry<'a, P> {
    fn new(mu: f64, weight: f64, sigma: f64) -> Self {
        Entry::Sample { mu, weight, sigma }
    }

    fn mu(&self) -> f64 {
        match *self {
            Entry::Sample { mu, .. } | Entry::Prior { mu, .. } => mu,
        }
    }

    fn weight(&self) -> f64 {
        match *self {
            Entry::Sample { weight, .. } | Entry::Prior { weight, .. } => weight,
        }
    }

    fn set_weight(&mut self, w: f64) {
        match self {
            Entry::Sample { weight, .. } | Entry::Prior { weight, .. } => {
                *weight = w;
            }
        }
    }

    fn set_sigma(&mut self, v: f64) {
        match self {
            Entry::Sample { sigma, .. } => {
                *sigma = v;
            }
            _ => {}
        }
    }
}
impl<'a, P> Entry<'a, P>
where
    P: Numerical + PriorCdf + PriorPdf,
{
    fn cdf(&self, x: f64) -> f64 {
        match *self {
            Entry::Sample { mu, sigma, .. } => {
                let dist = Normal::new(mu, sigma).unwrap_or_else(|e| unreachable!("{}", e));
                dist.cdf(x)
            }
            Entry::Prior { param_space, .. } => param_space.cdf(x),
        }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        match *self {
            Entry::Sample { mu, sigma, .. } => {
                let dist = Normal::new(mu, sigma).unwrap_or_else(|e| unreachable!("{}", e));
                dist.ln_pdf(x)
            }
            Entry::Prior { param_space, .. } => param_space.ln_pdf(x),
        }
    }
}
