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
pub struct ParzenEstimator<W> {
    params: ParzenEstimatorParameters<W>,
}
impl<W: Weights> ParzenEstimator<W> {
    pub fn new(params: ParzenEstimatorParameters<W>) -> Self {
        Self { params }
    }
}
impl<W: Weights> ParzenEstimator<W> {
    pub fn estimate(&self, mus: &[f64], low: f64, high: f64) -> Estimated {
        let weights = self.params.weights.weights(mus.len());
        let mut weighted_mus = mus
            .iter()
            .cloned()
            .zip(weights.into_iter())
            .map(|(mu, weight)| WeightedMu { mu, weight })
            .collect::<Vec<_>>();
        weighted_mus.sort_by_key(|x| NonNanF64::new(x.mu));

        let mut prior_pos = None;
        if self.params.consider_prior {
            let prior_mu = 0.5 * (low + high);
            if mus.is_empty() {
                weighted_mus.push(WeightedMu {
                    mu: prior_mu,
                    weight: self.params.prior_weight,
                });
            } else {
                // We decide the place of the  prior.
                let pos = weighted_mus
                    .binary_search_by(|x| {
                        if NonNanF64::new(prior_mu) == NonNanF64::new(x.mu) {
                            ::std::cmp::Ordering::Greater
                        } else {
                            NonNanF64::new(x.mu).cmp(&NonNanF64::new(prior_mu))
                        }
                    })
                    .unwrap_or_else(|i| i);
                weighted_mus.insert(
                    pos,
                    WeightedMu {
                        mu: prior_mu,
                        weight: self.params.prior_weight,
                    },
                );
                prior_pos = Some(pos);
            }
        }

        let weight_sum = weighted_mus.iter().map(|x| x.weight).sum::<f64>();
        for x in &mut weighted_mus {
            x.weight /= weight_sum;
        }

        let mut sigma: Vec<f64> = Vec::new();
        if !mus.is_empty() {
            use std::iter::once;

            for ((prev, curr), succ) in once(low)
                .chain(weighted_mus.iter().map(|x| x.mu))
                .zip(weighted_mus.iter().map(|x| x.mu))
                .zip(weighted_mus.iter().map(|x| x.mu).skip(1).chain(once(high)))
            {
                sigma.push(float::max(curr - prev, succ - curr));
            }
            let n = sigma.len();
            if !self.params.consider_endpoints {
                sigma[0] = sigma[1] - sigma[0];
                sigma[n - 1] = sigma[n - 1] - sigma[n - 2];
            }
        } else if self.params.consider_prior {
            let prior_sigma = 1.0 * (high - low);
            sigma.push(prior_sigma);
        }

        // Adjust the range of the `sigma` according to the `consider_magic_clip` flag.
        let maxsigma = 1.0 * (high - low);
        let minsigma;
        if self.params.consider_magic_clip {
            minsigma = 1.0 * (high - low) / float::min(100.0, 1.0 + (weighted_mus.len() as f64));
        } else {
            minsigma = 0.0;
        }
        for s in &mut sigma {
            *s = float::clip(minsigma, *s, maxsigma);
        }
        if let Some(pos) = prior_pos {
            let prior_sigma = 1.0 * (high - low);
            sigma[pos] = prior_sigma;
        }

        Estimated {
            mus: weighted_mus,
            sigmas: sigma,
        }
    }
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
        let estimator = ParzenEstimator::new(params);
        let result = estimator.estimate(&[], 0.0, 1.0);

        assert_eq!(result.sigmas, [1.0]);
        assert_eq!(result.mus, [m(0.5, 1.0)]);
    }

    #[test]
    fn it_works1() {
        let params = ParzenEstimatorParameters::default();
        let estimator = ParzenEstimator::new(params);
        let result = estimator.estimate(&[2.4, 3.3], 0.0, 1.0);

        assert_eq!(result.sigmas, [1.0, 1.0, 0.25]);
        assert_eq!(
            result.mus,
            [
                m(0.5, 0.3333333333333333),
                m(2.4, 0.3333333333333333),
                m(3.3, 0.3333333333333333),
            ]
        );
    }

    #[test]
    fn it_works2() {
        let estimator = ParzenEstimator::new(ParzenEstimatorParameters::default());
        let x = estimator.estimate(&[3.], 0.5, 3.5);
        assert_eq!(weights(&x), [0.5, 0.5]);
        assert_eq!(mus(&x), [2.0, 3.0]);
        assert_eq!(x.sigmas, [3.0, 1.5]);
    }
    #[test]
    fn it_works3() {
        let estimator = ParzenEstimator::new(ParzenEstimatorParameters::default());
        let x = estimator.estimate(&[3., 1., 3., 3., 3., 1., 2., 2., 2.], 0.5, 3.5);
        assert_eq!(
            weights(&x),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        );
        assert_eq!(mus(&x), [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);

        assert_eq!(
            x.sigmas,
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
        let estimator = ParzenEstimator::new(ParzenEstimatorParameters::default());
        let x = estimator.estimate(&[1.95032376], 1.3862943611198906, 4.852030263919617);
        assert_eq!(weights(&x), [0.5, 0.5]);
        assert_eq!(mus(&x), [1.95032376, 3.119162312519754]);
        assert_eq!(x.sigmas, [1.155245300933242, 3.465735902799726]);
    }
    #[test]
    fn it_works5() {
        let estimator = ParzenEstimator::new(ParzenEstimatorParameters::default());
        let x = estimator.estimate(
            &[
                1.53647634, 1.60117829, 1.74975032, 3.78253979, 3.75193948, 4.77576884, 1.64391653,
                4.18670963, 3.40994179,
            ],
            1.3862943611198906,
            4.852030263919617,
        );
        assert_eq!(
            weights(&x),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        );
        assert_eq!(
            mus(&x),
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
            x.sigmas,
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
        let estimator = ParzenEstimator::new(ParzenEstimatorParameters::default());
        let x = estimator.estimate(&[-11.94114835], -23.025850929940457, -6.907755278982137);
        assert_eq!(weights(&x), [0.5, 0.5]);
        assert_eq!(mus(&x), [-14.966803104461297, -11.94114835]);
        assert_eq!(x.sigmas, [16.11809565095832, 8.05904782547916]);
    }
    #[test]
    fn it_works7() {
        let estimator = ParzenEstimator::new(ParzenEstimatorParameters::default());
        let x = estimator.estimate(
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
            weights(&x),
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        );
        assert_eq!(
            mus(&x),
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
            x.sigmas,
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

    fn m(mu: f64, weight: f64) -> WeightedMu {
        WeightedMu { mu, weight }
    }

    fn mus(x: &Estimated) -> Vec<f64> {
        x.mus.iter().map(|x| x.mu).collect()
    }

    fn weights(x: &Estimated) -> Vec<f64> {
        x.mus.iter().map(|x| x.weight).collect()
    }
}
