use crate::float::NonNanF64;
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

        if self.params.consider_prior {
            let prior_mu = 0.5 * (low + high);
            if mus.is_empty() {
                weighted_mus.push(WeightedMu {
                    mu: prior_mu,
                    weight: self.params.prior_weight,
                });
            } else {
                // We decide the place of the  prior.
                let prior_pos = weighted_mus
                    .binary_search_by_key(&NonNanF64::new(prior_mu), |x| NonNanF64::new(x.mu))
                    .unwrap_or_else(|i| i);
                weighted_mus.insert(
                    prior_pos,
                    WeightedMu {
                        mu: prior_mu,
                        weight: self.params.prior_weight,
                    },
                );
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
                sigma.push(fmax(curr - prev, succ - prev));
            }
            let n = sigma.len();
            if !self.params.consider_endpoints {
                sigma[0] = sigma[1] - sigma[0];
                sigma[n - 2] = sigma[n - 2] - sigma[n - 3];
            }
            sigma.truncate(n - 1);
        } else if self.params.consider_prior {
            let prior_sigma = 1.0 * (high - low);
            sigma.push(prior_sigma);
        }

        // Adjust the range of the `sigma` according to the `consider_magic_clip` flag.
        panic!()
    }
}

fn fmax(x: f64, y: f64) -> f64 {
    cmp::max(NonNanF64::new(x), NonNanF64::new(y)).as_f64()
}

#[derive(Debug)]
pub struct WeightedMu {
    pub mu: f64,
    pub weight: f64,
}

#[derive(Debug)]
pub struct Estimated {
    pub weights: Vec<f64>,
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
}
