use crate::float::{self, NonNanF64};
use rand;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp;
use std::f64::EPSILON;

// TODO: s/Builder/Options/
#[derive(Debug)]
pub struct ParzenEstimatorBuilder {
    consider_prior: bool,
    prior_weight: f64,
    consider_magic_clip: bool,
    consider_endpoints: bool,
}
impl ParzenEstimatorBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn consider_prior(&mut self, flag: bool) -> &mut Self {
        self.consider_prior = flag;
        self
    }

    pub fn prior_weight(&mut self, weight: f64) -> &mut Self {
        self.prior_weight = weight;
        self
    }

    pub fn consider_magic_clip(&mut self, flag: bool) -> &mut Self {
        self.consider_magic_clip = flag;
        self
    }

    pub fn consider_endpoints(&mut self, flag: bool) -> &mut Self {
        self.consider_endpoints = flag;
        self
    }

    pub fn finish<M, W>(&self, mus: M, weights: W, low: f64, high: f64) -> ParzenEstimator
    where
        M: Iterator<Item = f64>,
        W: Iterator<Item = f64>,
    {
        let mut entries = self.make_sorted_entries(mus, weights);

        let prior_pos = if self.consider_prior {
            // TODO: `|| is_empty()`
            let prior_mu = 0.5 * (low + high);
            let inserted_pos = self.insert_prior_entry(&mut entries, prior_mu);
            Some(inserted_pos)
        } else {
            None
        };

        self.normalize_weights(&mut entries);

        self.setup_sigmas(&mut entries, low, high);
        if let Some(pos) = prior_pos {
            let prior_sigma = high - low;
            entries[pos].sigma = prior_sigma;
        }

        let p_accept = entries
            .iter()
            .map(|e| (e.normal_cdf(high) - e.normal_cdf(low)) * e.weight)
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
            .map(|(mu, weight)| Entry::new(mu, weight))
            .collect::<Vec<_>>();
        entries.sort_by_key(|x| NonNanF64::new(x.mu));
        entries
    }

    fn insert_prior_entry(&self, entries: &mut Vec<Entry>, prior_mu: f64) -> usize {
        let pos = entries
            .binary_search_by(|x| {
                NonNanF64::new(x.mu)
                    .cmp(&NonNanF64::new(prior_mu))
                    .then(cmp::Ordering::Greater)
            })
            .unwrap_or_else(|i| i);
        entries.insert(pos, Entry::new(prior_mu, self.prior_weight));
        pos
    }

    fn normalize_weights(&self, entries: &mut [Entry]) {
        let weight_sum = entries.iter().map(|x| x.weight).sum::<f64>();
        for x in entries {
            x.weight /= weight_sum;
        }
    }

    fn setup_sigmas(&self, entries: &mut [Entry], low: f64, high: f64) {
        assert!(low < high, "low={}, high={}", low, high);

        for i in 0..entries.len() {
            let prev = if i == 0 { low } else { entries[i - 1].mu };
            let curr = entries[i].mu;
            let succ = entries.get(i + 1).map_or(high, |x| x.mu);
            entries[i].sigma = float::max(curr - prev, succ - curr);
        }

        if !self.consider_endpoints {
            let n = entries.len();
            if n >= 2 {
                entries[0].sigma = entries[1].mu - entries[0].mu;
                entries[n - 1].sigma = entries[n - 1].mu - entries[n - 2].mu;
            }
        }

        let maxsigma = high - low;
        let minsigma = if self.consider_magic_clip {
            (high - low) / float::min(100.0, 1.0 + (entries.len() as f64))
        } else {
            EPSILON
        };
        for x in entries {
            x.sigma = float::clip(minsigma, x.sigma, maxsigma);
        }
    }
}
impl Default for ParzenEstimatorBuilder {
    fn default() -> Self {
        Self {
            consider_prior: true,
            prior_weight: 1.0,
            consider_magic_clip: true,
            consider_endpoints: false,
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
    pub fn mus<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.mu)
    }

    pub fn weights<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.weight)
    }

    pub fn sigmas<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.sigma)
    }

    pub fn gmm(&self) -> Gmm {
        Gmm { estimator: self }
    }
}

fn logsumexp(xs: &[f64]) -> f64 {
    let max_x = xs.iter().max_by_key(|&&x| NonNanF64::new(x)).expect("TODO");
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
            let x = log_density + (e.weight / self.estimator.p_accept).ln();
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
                .choose_weighted(rng, |x| x.weight)
                .expect("TODO");
            let d = rand::distributions::Normal::new(entry.mu, entry.sigma);
            let draw = d.sample(rng);
            if self.estimator.low <= draw && draw < self.estimator.high {
                return draw;
            }
        }
    }
}

#[derive(Debug)]
pub struct Entry {
    pub mu: f64,
    pub weight: f64,
    pub sigma: f64, // std-dev
}
impl Entry {
    fn new(mu: f64, weight: f64) -> Self {
        Entry {
            mu,
            weight,
            sigma: 0.0,
        }
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        use statrs::distribution::{Normal, Univariate};
        Normal::new(self.mu, self.sigma).expect("TODO").cdf(x)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        use statrs::distribution::{Continuous, Normal};
        Normal::new(self.mu, self.sigma).expect("TODO").ln_pdf(x)
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
        ParzenEstimatorBuilder::new().finish(mus.iter().cloned(), weights, low, high)
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
        assert_eq!(est.sigmas().collect::<Vec<_>>(), [1.0, 1.0, 0.25]);
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
        assert_eq!(est.sigmas().collect::<Vec<_>>(), [3.0, 1.5]);
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
                0.5,
                1.0,
                3.0,
                0.2727272727272727,
                0.2727272727272727,
                1.0,
                1.0,
                0.2727272727272727,
                0.2727272727272727,
                0.5
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
            [1.155245300933242, 3.465735902799726]
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
                0.31506690025452055
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
            [16.11809565095832, 8.05904782547916]
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
                1.4652814228143927,
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
