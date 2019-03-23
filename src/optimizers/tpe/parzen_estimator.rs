use crate::float::{self, NonNanF64};
use rand;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use statrs::distribution::{Continuous, Normal, Univariate};
use std::cmp;

#[derive(Debug)]
pub struct ParzenEstimatorBuilder {
    prior_weight: f64,
    prior_uniform: bool,
}
impl ParzenEstimatorBuilder {
    pub fn new(prior_weight: f64, prior_uniform: bool) -> Self {
        Self {
            prior_weight,
            prior_uniform,
        }
    }

    pub fn finish<M, W>(&self, mus: M, weights: W, low: f64, high: f64) -> ParzenEstimator
    where
        M: Iterator<Item = f64>,
        W: Iterator<Item = f64>,
    {
        let mut entries = self.make_sorted_entries(mus, weights);
        let prior_mu = 0.5 * (low + high);
        let prior_sigma = high - low;
        self.insert_prior_entry(&mut entries, prior_mu, prior_sigma, low, high);

        self.normalize_weights(&mut entries);
        self.setup_sigmas(&mut entries, low, high);

        let p_accept = entries
            .iter()
            .map(|e| (e.normal_cdf(high) - e.normal_cdf(low)) * e.weight())
            .sum::<f64>();

        ParzenEstimator {
            entries,
            low,
            high,
            p_accept,
        }
    }

    fn make_sorted_entries<M, W>(&self, mus: M, weights: W) -> Vec<Entry>
    where
        M: Iterator<Item = f64>,
        W: Iterator<Item = f64>,
    {
        let mut entries = mus
            .zip(weights)
            .map(|(mu, weight)| Entry::new(mu, weight, 0.0, false))
            .collect::<Vec<_>>();
        entries.sort_by_key(|x| NonNanF64::new(x.mu()));
        entries
    }

    fn insert_prior_entry(
        &self,
        entries: &mut Vec<Entry>,
        prior_mu: f64,
        prior_sigma: f64,
        low: f64,
        high: f64,
    ) {
        let pos = entries
            .binary_search_by(|x| {
                NonNanF64::new(x.mu())
                    .cmp(&NonNanF64::new(prior_mu))
                    .then(cmp::Ordering::Greater)
            })
            .unwrap_or_else(|i| i);
        let entry = if self.prior_uniform {
            Entry::Uniform {
                mu: prior_mu,
                weight: self.prior_weight,
                low,
                high,
            }
        } else {
            Entry::new(prior_mu, self.prior_weight, prior_sigma, true)
        };
        entries.insert(pos, entry);
    }

    fn normalize_weights(&self, entries: &mut [Entry]) {
        let weight_sum = entries.iter().map(|x| x.weight()).sum::<f64>();
        for x in entries {
            let w = x.weight() / weight_sum;
            x.set_weight(w);
        }
    }

    fn setup_sigmas(&self, entries: &mut [Entry], low: f64, high: f64) {
        assert!(low < high, "low={}, high={}", low, high);

        for i in 0..entries.len() {
            let prev = if i == 0 { low } else { entries[i - 1].mu() };
            let curr = entries[i].mu();
            let succ = entries.get(i + 1).map_or(high, |x| x.mu());
            entries[i].set_sigma(float::max(curr - prev, succ - curr));
        }

        let n = entries.len();
        if n >= 2 {
            entries[0].set_sigma(entries[1].mu() - entries[0].mu());
            entries[n - 1].set_sigma(entries[n - 1].mu() - entries[n - 2].mu());
        }

        let maxsigma = high - low;
        let minsigma = (high - low) / float::min(100.0, 1.0 + (entries.len() as f64));
        for x in entries {
            if let Some(sigma) = x.sigma() {
                x.set_sigma(float::clip(minsigma, sigma, maxsigma));
            }
        }
    }
}
impl Default for ParzenEstimatorBuilder {
    fn default() -> Self {
        Self {
            prior_weight: 1.0,
            prior_uniform: false,
        }
    }
}

#[derive(Debug)]
pub struct ParzenEstimator {
    entries: Vec<Entry>,
    low: f64,
    high: f64,
    p_accept: f64,
}
impl ParzenEstimator {
    pub fn gmm(&self) -> Gmm {
        Gmm { estimator: self }
    }

    #[cfg(test)]
    pub fn mus<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.mu())
    }

    #[cfg(test)]
    pub fn weights<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.weight())
    }

    #[cfg(test)]
    pub fn sigmas<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.sigma().expect("TODO"))
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
pub struct Gmm<'a> {
    estimator: &'a ParzenEstimator,
}
impl<'a> Gmm<'a> {
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
impl<'a> Distribution<f64> for Gmm<'a> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        loop {
            let entry = self
                .estimator
                .entries
                .choose_weighted(rng, |x| x.weight())
                .expect("never fails");
            match *entry {
                Entry::Normal { mu, sigma, .. } => {
                    let d = rand::distributions::Normal::new(mu, sigma);
                    let draw = d.sample(rng);
                    if self.estimator.low <= draw && draw < self.estimator.high {
                        return draw;
                    }
                }
                Entry::Uniform { low, high, .. } => {
                    let d = rand::distributions::Uniform::new(low, high);
                    return d.sample(rng);
                }
            }
        }
    }
}

#[derive(Debug)]
enum Entry {
    Normal {
        mu: f64,
        weight: f64,
        sigma: f64,  // std-dev
        prior: bool, // TODO: remove
    },
    Uniform {
        mu: f64, // TODO: delete
        weight: f64,
        low: f64,
        high: f64,
    },
}
impl Entry {
    fn new(mu: f64, weight: f64, sigma: f64, prior: bool) -> Self {
        Entry::Normal {
            mu,
            weight,
            sigma,
            prior,
        }
    }

    fn mu(&self) -> f64 {
        match *self {
            Entry::Normal { mu, .. } | Entry::Uniform { mu, .. } => mu,
        }
    }

    fn weight(&self) -> f64 {
        match *self {
            Entry::Normal { weight, .. } | Entry::Uniform { weight, .. } => weight,
        }
    }

    fn set_weight(&mut self, w: f64) {
        match self {
            Entry::Normal { weight, .. } | Entry::Uniform { weight, .. } => {
                *weight = w;
            }
        }
    }

    fn sigma(&self) -> Option<f64> {
        match *self {
            Entry::Normal { sigma, .. } => Some(sigma),
            _ => None,
        }
    }

    fn set_sigma(&mut self, v: f64) {
        match self {
            Entry::Normal { sigma, prior, .. } if !*prior => {
                *sigma = v;
            }
            _ => {}
        }
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        match *self {
            Entry::Normal { mu, sigma, .. } => Normal::new(mu, sigma).expect("never fails").cdf(x),
            Entry::Uniform { low, high, .. } => (x - low) / (high - low),
        }
    }

    fn log_pdf(&self, x: f64) -> f64 {
        match *self {
            Entry::Normal { mu, sigma, .. } => {
                Normal::new(mu, sigma).expect("never fails").ln_pdf(x)
            }
            Entry::Uniform { low, high, .. } => 1f64.ln() - (high - low).ln(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iter::linspace;
    use std::iter::repeat;

    fn estimator(mus: &[f64], low: f64, high: f64) -> ParzenEstimator {
        let n = mus.len();
        let m = cmp::max(n, 25) - 25;
        let weights = linspace(1.0 / (n as f64), 1.0, m).chain(repeat(1.0).take(n - m));
        ParzenEstimatorBuilder::new(1.0, false).finish(mus.iter().cloned(), weights, low, high)
    }

    #[test]
    fn it_works0() {
        let est = estimator(&[], 0.0, 1.0);
        assert_eq!(est.sigmas().collect::<Vec<_>>(), [1.0]);
        assert_eq!(est.mus().collect::<Vec<_>>(), [0.5]);
        assert_eq!(est.weights().collect::<Vec<_>>(), [1.0]);
    }

    #[test]
    fn it_works1() {
        let est = estimator(&[2.4, 3.3], 0.0, 1.0);
        assert_eq!(
            est.sigmas().collect::<Vec<_>>(),
            [1.0, 1.0, 0.8999999999999999]
        );
        assert_eq!(est.mus().collect::<Vec<_>>(), [0.5, 2.4, 3.3]);
        assert_eq!(
            est.weights().collect::<Vec<_>>(),
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
        );
    }

    #[test]
    fn it_works2() {
        let est = estimator(&[3.], 0.5, 3.5);
        assert_eq!(est.weights().collect::<Vec<_>>(), [0.5, 0.5]);
        assert_eq!(est.mus().collect::<Vec<_>>(), [2.0, 3.0]);
        assert_eq!(est.sigmas().collect::<Vec<_>>(), [3.0, 1.0]);
    }

    #[test]
    fn it_works3() {
        let est = estimator(&[3., 1., 3., 3., 3., 1., 2., 2., 2.], 0.5, 3.5);
        assert_eq!(
            est.weights().collect::<Vec<_>>(),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        );
        assert_eq!(
            est.mus().collect::<Vec<_>>(),
            [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
        assert_eq!(
            est.sigmas().collect::<Vec<_>>(),
            [
                0.2727272727272727,
                1.0,
                3.0,
                0.2727272727272727,
                0.2727272727272727,
                1.0,
                1.0,
                0.2727272727272727,
                0.2727272727272727,
                0.2727272727272727,
            ]
        );
    }

    #[test]
    fn it_works4() {
        let est = estimator(&[1.95032376], 1.3862943611198906, 4.852030263919617);
        assert_eq!(est.weights().collect::<Vec<_>>(), [0.5, 0.5]);
        assert_eq!(
            est.mus().collect::<Vec<_>>(),
            [1.95032376, 3.119162312519754]
        );
        assert_eq!(
            est.sigmas().collect::<Vec<_>>(),
            [1.1688385525197538, 3.465735902799726]
        );
    }

    #[test]
    fn it_works5() {
        let est = estimator(
            &[
                1.53647634, 1.60117829, 1.74975032, 3.78253979, 3.75193948, 4.77576884, 1.64391653,
                4.18670963, 3.40994179,
            ],
            1.3862943611198906,
            4.852030263919617,
        );
        assert_eq!(
            est.weights().collect::<Vec<_>>(),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        );
        assert_eq!(
            est.mus().collect::<Vec<_>>(),
            [
                1.53647634,
                1.60117829,
                1.64391653,
                1.74975032,
                3.119162312519754,
                3.40994179,
                3.75193948,
                3.78253979,
                4.18670963,
                4.77576884,
            ]
        );
        assert_eq!(
            est.sigmas().collect::<Vec<_>>(),
            [
                0.31506690025452055,
                0.31506690025452055,
                0.31506690025452055,
                1.3694119925197539,
                3.465735902799726,
                0.3419976899999999,
                0.3419976899999999,
                0.4041698400000002,
                0.5890592099999994,
                0.5890592099999994
            ]
        );
    }

    #[test]
    fn it_works6() {
        let est = estimator(&[-11.94114835], -23.025850929940457, -6.907755278982137);
        assert_eq!(est.weights().collect::<Vec<_>>(), [0.5, 0.5]);
        assert_eq!(
            est.mus().collect::<Vec<_>>(),
            [-14.966803104461297, -11.94114835]
        );
        assert_eq!(
            est.sigmas().collect::<Vec<_>>(),
            [16.11809565095832, 5.37269855031944]
        );
    }

    #[test]
    fn it_works7() {
        let est = estimator(
            &[
                -7.26690481,
                -7.78043504,
                -22.90676614,
                -12.96005192,
                -19.10557622,
                -13.05687971,
                -15.45543074,
                -9.98658409,
                -10.75822351,
            ],
            -23.025850929940457,
            -6.907755278982137,
        );
        assert_eq!(
            est.weights().collect::<Vec<_>>(),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        );
        assert_eq!(
            est.mus().collect::<Vec<_>>(),
            [
                -22.90676614,
                -19.10557622,
                -15.45543074,
                -14.966803104461297,
                -13.05687971,
                -12.96005192,
                -10.75822351,
                -9.98658409,
                -7.78043504,
                -7.26690481
            ]
        );
        assert_eq!(
            est.sigmas().collect::<Vec<_>>(),
            [
                3.801189919999999,
                3.801189919999999,
                3.650145479999999,
                16.11809565095832,
                1.9099233944612966,
                2.201828409999999,
                2.201828409999999,
                2.2061490499999987,
                2.2061490499999987,
                1.4652814228143927
            ]
        );
    }
}
