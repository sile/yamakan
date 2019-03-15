use crate::iter::linspace;
use crate::optimizer::Observation;
use std::cmp;
use std::iter::repeat;

pub use self::categorical::{TpeCategoricalOptimizer, TpeCategoricalOptions};
pub use self::numerical::TpeNumericalOptimizer;
pub use self::parzen_estimator::ParzenEstimatorBuilder; // TODO

mod categorical;
mod numerical;
mod parzen_estimator;

pub trait Preprocess<P, V> {
    fn divide_observations(&self, observations: &[Observation<P, V>]) -> usize;
    fn weight_observations(
        &self,
        observations: &[Observation<P, V>],
        is_superior: bool,
    ) -> Box<dyn Iterator<Item = f64>>;
}

#[derive(Debug, Default)]
pub struct DefaultPreprocessor;
impl<P, V> Preprocess<P, V> for DefaultPreprocessor {
    fn divide_observations(&self, observations: &[Observation<P, V>]) -> usize {
        let n = observations.len() as f64;
        cmp::min((0.25 * n.sqrt()).ceil() as usize, 25)
    }

    fn weight_observations(
        &self,
        observations: &[Observation<P, V>],
        is_superior: bool,
    ) -> Box<dyn Iterator<Item = f64>> {
        let n = observations.len();
        if is_superior {
            Box::new(repeat(1.0).take(n))
        } else {
            let m = cmp::max(n, 25) - 25;
            Box::new(linspace(1.0 / (n as f64), 1.0, m).chain(repeat(1.0).take(n - m)))
        }
    }
}

// TODO: delete
pub trait TpeStrategy<P, V> {
    fn divide_observations<'a>(&self, observations: &'a [Observation<P, V>]) -> usize;
    fn weight_superiors(&self, superiors: &[Observation<P, V>]) -> Vec<f64>;
    fn weight_inferiors(&self, inferiors: &[Observation<P, V>]) -> Vec<f64>;
}

#[derive(Debug, Default)]
pub struct DefaultTpeStrategy;
impl<P, V> TpeStrategy<P, V> for DefaultTpeStrategy {
    fn divide_observations<'a>(&self, observations: &'a [Observation<P, V>]) -> usize {
        let n = observations.len() as f64;
        cmp::min((0.25 * n.sqrt()).ceil() as usize, 25)
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
