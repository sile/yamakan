use crate::iter::linspace;
use std::cmp;
use std::iter::repeat;

pub use self::categorical::{TpeCategoricalOptimizer, TpeCategoricalOptimizerBuilder};
pub use self::numerical::TpeNumericalOptimizer;
pub use self::parzen_estimator::ParzenEstimatorBuilder; // TODO

mod categorical;
mod numerical;
mod parzen_estimator;

#[derive(Debug)]
pub struct Observation<P, V> {
    pub param: P,
    pub value: V,
}

pub trait TpeStrategy<P, V> {
    fn divide_observations<'a>(
        &self,
        observations: &'a [Observation<P, V>],
    ) -> (&'a [Observation<P, V>], &'a [Observation<P, V>]);
    fn weight_superiors(&self, superiors: &[Observation<P, V>]) -> Vec<f64>;
    fn weight_inferiors(&self, inferiors: &[Observation<P, V>]) -> Vec<f64>;
}

#[derive(Debug, Default)]
pub struct DefaultTpeStrategy;
impl<P, V> TpeStrategy<P, V> for DefaultTpeStrategy {
    fn divide_observations<'a>(
        &self,
        observations: &'a [Observation<P, V>],
    ) -> (&'a [Observation<P, V>], &'a [Observation<P, V>]) {
        let n = observations.len() as f64;
        let gamma = cmp::min((0.25 * n.sqrt()).ceil() as usize, 25);
        observations.split_at(gamma)
    }

    fn weight_superiors(&self, superiors: &[Observation<P, V>]) -> Vec<f64> {
        vec![1.0; superiors.len()]
    }

    fn weight_inferiors(&self, inferiors: &[Observation<P, V>]) -> Vec<f64> {
        let n = inferiors.len();
        let m = cmp::max(n, 25) - 25;
        linspace(1.0 / (n as f64), 1.0, m)
            .chain(repeat(1.0).take(n - m))
            .collect()
    }
}
