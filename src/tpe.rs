use crate::iter::linspace;
use std::cmp;

#[derive(Debug)]
pub struct ParzenEstimatorParameters<W> {
    pub consider_prior: bool,
    pub prior_weight: Option<f64>,
    pub consider_magic_clip: bool,
    pub consider_endpoints: bool,
    pub weights: W,
}
impl Default for ParzenEstimatorParameters<DefaultWeights> {
    fn default() -> Self {
        Self {
            consider_prior: true,
            prior_weight: Some(1.0),
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
        panic!()
    }
}

#[derive(Debug)]
pub struct Estimated {
    pub weights: Vec<f64>,
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
}
