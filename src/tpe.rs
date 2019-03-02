use self::parzen_estimator::ParzenEstimatorBuilder;
use crate::float::NonNanF64;
use crate::iter::linspace;
use std::cmp;
use std::iter::repeat;

pub mod parzen_estimator;

pub fn default_weights(mus_len: usize) -> impl Iterator<Item = f64> {
    let n = cmp::max(mus_len, 25) - 25;
    linspace(1.0 / (mus_len as f64), 1.0, n).chain(repeat(1.0).take(mus_len - n))
}

pub fn default_gamma(mus_len: usize) -> usize {
    let n = mus_len as f64;
    cmp::min((0.25 * n.sqrt()).ceil() as usize, 25)
}

const EI_CANDIDATES: usize = 24;

#[derive(Debug)]
pub struct TpeOptimizer {
    estimator_builder: ParzenEstimatorBuilder,
    observations: Vec<Observation>,
}
impl TpeOptimizer {
    pub fn new() -> Self {
        let estimator_builder = ParzenEstimatorBuilder::new();
        Self {
            estimator_builder,
            observations: Vec::new(),
        }
    }

    pub fn tell(&mut self, x: f64, y: f64) {
        let o = Observation::new(x, y);
        let i = self
            .observations
            .binary_search_by_key(&NonNanF64::new(o.y), |t| NonNanF64::new(t.y))
            .unwrap_or_else(|i| i);
        self.observations.insert(i, o);
    }

    pub fn ask_uniform(&self, low: f64, high: f64) -> Option<f64> {
        let (below, above) = self.split_observations();
        if below.is_empty() || above.is_empty() {
            return None;
        }

        let estimator_below = self.estimator_builder.finish(
            below.iter().map(|o| o.x),
            default_weights(below.len()),
            low,
            high,
        );
        let samples_below = estimator_below
            .samples_from_gmm()
            .take(EI_CANDIDATES)
            .collect::<Vec<_>>();
        let log_likelihoods_below = estimator_below.gmm_log_pdf(&samples_below);

        let estimator_above = self.estimator_builder.finish(
            above.iter().map(|o| o.x),
            default_weights(above.len()),
            low,
            high,
        );
        let log_likelihoods_above = estimator_above.gmm_log_pdf(&samples_below);

        Some(self.compare(
            &samples_below,
            &log_likelihoods_below,
            &log_likelihoods_above,
        ))
    }

    pub fn ask_categorical(&self) -> usize {
        panic!()
    }

    fn split_observations(&self) -> (&[Observation], &[Observation]) {
        let below_num = default_gamma(self.observations.len()); // TODO
        self.observations.split_at(below_num)
    }

    // TODO: rename
    fn compare(&self, samples: &[f64], log_l: &[f64], log_g: &[f64]) -> f64 {
        log_l
            .iter()
            .zip(log_g.iter())
            .zip(samples.iter())
            .max_by_key(|((&l, &g), _)| NonNanF64::new(l - g))
            .map(|(_, &s)| s)
            .expect("TODO")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub x: f64,
    pub y: f64,
}
impl Observation {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut opt = TpeOptimizer::new();
        opt.tell(1.0, 0.1);
        opt.tell(2.0, 0.3);
        opt.tell(2.5, 1.0);
        opt.tell(3.0, 0.5);
        assert_eq!(opt.ask_uniform(0.0, 5.0), Some(0.1)); // TODO
    }
}
