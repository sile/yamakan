use crate::float::{self, NonNanF64};
use crate::iter::linspace;
use std::cmp;

#[derive(Debug)]
pub struct ParzenEstimatorParameters<W> {
    pub consider_prior: bool,
    pub prior_weight: f64,
    pub consider_magic_clip: bool,
    pub consider_endpoints: bool,
    pub weights: W,
}
impl Default for ParzenEstimatorParameters<DefaultWeights> {
    fn default() -> Self {
        Self {
            consider_prior: true,
            prior_weight: 1.0,
            consider_magic_clip: true,
            consider_endpoints: false,
            weights: DefaultWeights,
        }
    }
}

pub trait Weights {
    fn weights(&self, x: usize) -> Vec<f64>;
}

#[derive(Debug)]
pub struct DefaultWeights;
impl Weights for DefaultWeights {
    fn weights(&self, x: usize) -> Vec<f64> {
        let mut weights = Vec::with_capacity(x);
        weights.extend(linspace(1.0 / (x as f64), 1.0, cmp::max(x, 25) - 25));
        for _ in weights.len()..x {
            weights.push(1.0);
        }
        weights
    }
}

#[derive(Debug)]
pub struct Entry {
    pub mu: f64,
    pub weight: f64,
    pub sigma: f64,
}
impl Entry {
    fn new(mu: f64, weight: f64) -> Self {
        Entry {
            mu,
            weight,
            sigma: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct ParzenEstimator<W> {
    params: ParzenEstimatorParameters<W>,
    entries: Vec<Entry>,
}
impl<W: Weights> ParzenEstimator<W> {
    pub fn new(params: ParzenEstimatorParameters<W>) -> Self {
        Self {
            params,
            entries: Vec::new(),
        }
    }

    pub fn mus<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.mu)
    }

    pub fn weights<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.weight)
    }

    pub fn sigmas<'a>(&'a self) -> impl Iterator<Item = f64> + 'a {
        self.entries.iter().map(|x| x.sigma)
    }
}
impl<W: Weights> ParzenEstimator<W> {
    fn insert_entry(entries: &mut Vec<Entry>, mu: f64, weight: f64) -> usize {
        let pos = entries
            .binary_search_by(|x| {
                NonNanF64::new(x.mu)
                    .cmp(&NonNanF64::new(mu))
                    .then(cmp::Ordering::Greater)
            })
            .unwrap_or_else(|i| i);
        entries.insert(pos, Entry::new(mu, weight));
        pos
    }

    pub fn estimate(&mut self, mus: &[f64], low: f64, high: f64) {
        let weights = self.params.weights.weights(mus.len());
        let mut entries = mus
            .iter()
            .zip(weights.into_iter())
            .map(|(&mu, weight)| Entry::new(mu, weight))
            .collect::<Vec<_>>();
        entries.sort_by_key(|x| NonNanF64::new(x.mu));

        let prior_pos = if self.params.consider_prior {
            let prior_mu = 0.5 * (low + high);
            let pos = Self::insert_entry(&mut entries, prior_mu, self.params.prior_weight);
            Some(pos)
        } else {
            None
        };

        let weight_sum = entries.iter().map(|x| x.weight).sum::<f64>();
        for x in &mut entries {
            x.weight /= weight_sum;
        }

        for i in 0..entries.len() {
            let prev = if i == 0 { low } else { entries[i - 1].mu };
            let curr = entries[i].mu;
            let succ = entries.get(i + 1).map_or(high, |x| x.mu);
            entries[i].sigma = float::max(curr - prev, succ - curr);
        }
        if !self.params.consider_endpoints {
            let n = entries.len();
            if n >= 2 {
                entries[0].sigma = entries[1].sigma - entries[0].sigma;
                entries[n - 1].sigma -= entries[n - 2].sigma;
            }
        }
        if let Some(pos) = prior_pos {
            let prior_sigma = high - low;
            entries[pos].sigma = prior_sigma;
        }

        // Adjust the range of the `sigma` according to the `consider_magic_clip` flag.
        let maxsigma = high - low;
        let minsigma = if self.params.consider_magic_clip {
            (high - low) / float::min(100.0, 1.0 + (entries.len() as f64))
        } else {
            0.0
        };
        for x in &mut entries {
            x.sigma = float::clip(minsigma, x.sigma, maxsigma);
        }

        self.entries = entries;
    }

    // pub fn estimate(&self, mus: &[f64], low: f64, high: f64) -> Estimated {
    //     let weights = self.params.weights.weights(mus.len());
    //     let mut weighted_mus = mus
    //         .iter()
    //         .cloned()
    //         .zip(weights.into_iter())
    //         .map(|(mu, weight)| WeightedMu { mu, weight })
    //         .collect::<Vec<_>>();
    //     weighted_mus.sort_by_key(|x| NonNanF64::new(x.mu));

    //     let mut prior_pos = None;
    //     if self.params.consider_prior {
    //         let prior_mu = 0.5 * (low + high);

    //         // We decide the place of the  prior.
    //         let pos = weighted_mus
    //             .binary_search_by(|x| {
    //                 NonNanF64::new(x.mu)
    //                     .cmp(&NonNanF64::new(prior_mu))
    //                     .then(cmp::Ordering::Greater)
    //             })
    //             .unwrap_or_else(|i| i);
    //         weighted_mus.insert(
    //             pos,
    //             WeightedMu {
    //                 mu: prior_mu,
    //                 weight: self.params.prior_weight,
    //             },
    //         );
    //         prior_pos = Some(pos);
    //     }

    //     let weight_sum = weighted_mus.iter().map(|x| x.weight).sum::<f64>();
    //     for x in &mut weighted_mus {
    //         x.weight /= weight_sum;
    //     }

    //     let mut sigma: Vec<f64> = Vec::new();
    //     if !mus.is_empty() {
    //         use std::iter::once;

    //         for ((prev, curr), succ) in once(low)
    //             .chain(weighted_mus.iter().map(|x| x.mu))
    //             .zip(weighted_mus.iter().map(|x| x.mu))
    //             .zip(weighted_mus.iter().map(|x| x.mu).skip(1).chain(once(high)))
    //         {
    //             sigma.push(float::max(curr - prev, succ - curr));
    //         }
    //         let n = sigma.len();
    //         if !self.params.consider_endpoints {
    //             sigma[0] = sigma[1] - sigma[0];
    //             sigma[n - 1] = sigma[n - 1] - sigma[n - 2];
    //         }
    //         if let Some(pos) = prior_pos {
    //             let prior_sigma = high - low;
    //             sigma[pos] = prior_sigma;
    //         }
    //     } else if self.params.consider_prior {
    //         let prior_sigma = high - low;
    //         sigma.push(prior_sigma);
    //     }

    //     // Adjust the range of the `sigma` according to the `consider_magic_clip` flag.
    //     let maxsigma = high - low;
    //     let minsigma = if self.params.consider_magic_clip {
    //         (high - low) / float::min(100.0, 1.0 + (weighted_mus.len() as f64))
    //     } else {
    //         0.0
    //     };
    //     for s in &mut sigma {
    //         *s = float::clip(minsigma, *s, maxsigma);
    //     }

    //     Estimated {
    //         mus: weighted_mus,
    //         sigmas: sigma,
    //     }
    // }
}

#[derive(Debug, PartialEq)]
pub struct WeightedMu {
    pub mu: f64,
    pub weight: f64,
}

#[derive(Debug, PartialEq)]
pub struct Estimated {
    pub mus: Vec<WeightedMu>,
    pub sigmas: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works0() {
        let params = ParzenEstimatorParameters::default();
        let mut est = ParzenEstimator::new(params);
        est.estimate(&[], 0.0, 1.0);

        assert_eq!(est.sigmas().collect::<Vec<_>>(), [1.0]);
        assert_eq!(est.mus().collect::<Vec<_>>(), [0.5]);
        assert_eq!(est.weights().collect::<Vec<_>>(), [1.0]);
    }

    #[test]
    fn it_works1() {
        let params = ParzenEstimatorParameters::default();
        let mut est = ParzenEstimator::new(params);
        est.estimate(&[2.4, 3.3], 0.0, 1.0);

        assert_eq!(est.sigmas().collect::<Vec<_>>(), [1.0, 1.0, 0.25]);
        assert_eq!(est.mus().collect::<Vec<_>>(), [0.5, 2.4, 3.3]);
        assert_eq!(
            est.weights().collect::<Vec<_>>(),
            [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
        );
    }

    #[test]
    fn it_works2() {
        let mut est = ParzenEstimator::new(ParzenEstimatorParameters::default());
        est.estimate(&[3.], 0.5, 3.5);
        assert_eq!(est.weights().collect::<Vec<_>>(), [0.5, 0.5]);
        assert_eq!(est.mus().collect::<Vec<_>>(), [2.0, 3.0]);
        assert_eq!(est.sigmas().collect::<Vec<_>>(), [3.0, 1.5]);
    }
    #[test]
    fn it_works3() {
        let mut est = ParzenEstimator::new(ParzenEstimatorParameters::default());
        est.estimate(&[3., 1., 3., 3., 3., 1., 2., 2., 2.], 0.5, 3.5);
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
        let mut est = ParzenEstimator::new(ParzenEstimatorParameters::default());
        est.estimate(&[1.95032376], 1.3862943611198906, 4.852030263919617);
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
        let mut est = ParzenEstimator::new(ParzenEstimatorParameters::default());
        est.estimate(
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
        let mut est = ParzenEstimator::new(ParzenEstimatorParameters::default());
        est.estimate(&[-11.94114835], -23.025850929940457, -6.907755278982137);
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
        let mut est = ParzenEstimator::new(ParzenEstimatorParameters::default());
        est.estimate(
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
