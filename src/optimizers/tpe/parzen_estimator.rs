use super::KdeStrategy;
use crate::float::NonNanF64;
use crate::range::Range;
use crate::spaces::{Numerical, PriorDistribution, PriorPdf};
use rand;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::distribution::{Continuous, Normal, Univariate};

#[derive(Debug)]
pub struct Sample {
    pub mu: f64, // TODO: rename (param?)
    pub weight: f64,
}
impl Sample {
    // TODO:
    fn cdf(&self, x: f64, bandwidth: f64) -> f64 {
        let dist = Normal::new(self.mu, bandwidth)
            .unwrap_or_else(|e| unreachable!("mu:{}, sd:{}, Error:{}", self.mu, bandwidth, e));
        dist.cdf(x)
    }

    // TODO:
    fn log_pdf(&self, x: f64, bandwidth: f64) -> f64 {
        let dist = Normal::new(self.mu, bandwidth).unwrap_or_else(|e| unreachable!("{}", e));
        dist.ln_pdf(x)
    }

    // TODO:
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, bandwidth: f64) -> f64 {
        rand::distributions::Normal::new(self.mu, bandwidth).sample(rng)
    }
}

#[derive(Debug)]
pub struct ParzenEstimatorBuilder<'a, P, S> {
    param_space: &'a P,
    strategy: &'a S,
    prior_weight: f64,
}
impl<'a, P, S> ParzenEstimatorBuilder<'a, P, S>
where
    P: Numerical,
    S: KdeStrategy,
{
    pub fn new(param_space: &'a P, strategy: &'a S, prior_weight: f64) -> Self {
        Self {
            param_space,
            strategy,
            prior_weight,
        }
    }

    pub fn finish<M, W>(&self, mus: M, weights: W) -> ParzenEstimator<P>
    where
        M: Iterator<Item = f64>,
        W: Iterator<Item = f64>,
    {
        let mut samples = mus
            .zip(weights)
            .map(|(mu, weight)| Sample { mu, weight })
            .collect::<Vec<_>>();
        let prior_weight = self.normalize_weights(&mut samples);

        let bandwidth = self
            .strategy
            .kde_bandwidth(&samples, self.param_space.range());

        let Range { low, high } = (*self.param_space).range(); // TODO:

        // TODO: prior-cdf
        let p_accept = samples
            .iter()
            .map(|s| (s.cdf(high, bandwidth) - s.cdf(low, bandwidth)) * s.weight)
            .sum::<f64>();

        ParzenEstimator {
            param_space: self.param_space,
            samples,
            bandwidth,
            p_accept,
            prior_weight,
        }
    }

    fn normalize_weights(&self, samples: &mut [Sample]) -> f64 {
        let mut weight_sum = samples.iter().map(|x| x.weight).sum::<f64>();
        for x in samples {
            x.weight /= weight_sum;
        }

        weight_sum += self.prior_weight;
        self.prior_weight / weight_sum
    }
}

#[derive(Debug)]
pub struct ParzenEstimator<'a, P> {
    param_space: &'a P,
    samples: Vec<Sample>,
    bandwidth: f64,
    p_accept: f64,
    prior_weight: f64,
}
impl<'a, P> ParzenEstimator<'a, P>
where
    P: PriorPdf,
{
    pub fn log_pdf(&self, param: f64) -> f64 {
        // TODO: use KDE
        let mut xs = Vec::with_capacity(self.samples.len());
        xs.push(self.param_space.ln_pdf(param));
        for s in &self.samples {
            let log_density = s.log_pdf(param, self.bandwidth);
            let x = log_density + (s.weight / self.p_accept).ln();
            xs.push(x);
        }
        logsumexp(&xs)
    }
}
impl<'a, P> Distribution<f64> for ParzenEstimator<'a, P>
where
    P: PriorDistribution + Numerical,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let x = if rng.gen_bool(self.prior_weight) {
                self.param_space
                    .to_f64(&self.param_space.sample(rng))
                    .expect("TODO")
            } else {
                let s = self
                    .samples
                    .choose_weighted(rng, |s| s.weight)
                    .unwrap_or_else(|e| unreachable!("{}", e));
                s.sample(rng, self.bandwidth)
            };
            if self.param_space.range().contains(&x) {
                return x;
            }
        }
    }
}

fn logsumexp(xs: &[f64]) -> f64 {
    let max_x = xs
        .iter()
        .max_by_key(|&&x| NonNanF64::new(x))
        .expect("never fails");
    xs.iter().map(|&x| (x - max_x).exp()).sum::<f64>().ln() + max_x
}
