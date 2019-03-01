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
                    .binary_search_by_key(&NonNanF64::new(prior_mu), |x| NonNanF64::new(x.mu))
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

    fn m(mu: f64, weight: f64) -> WeightedMu {
        WeightedMu { mu, weight }
    }
}
