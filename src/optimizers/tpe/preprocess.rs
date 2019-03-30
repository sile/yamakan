use crate::iter::linspace;
use crate::observation::Observation;
use std::cmp;
use std::iter::repeat;

pub trait Preprocess<P, V> {
    fn divide_observations(&self, observations: &[&Observation<P, V>]) -> usize;
    fn weight_observations(
        &self,
        observations: &[&Observation<P, V>],
        is_superior: bool,
    ) -> Box<dyn Iterator<Item = f64>>;
}

#[derive(Debug)]
pub struct DefaultPreprocessor {
    pub divide_factor: f64,
}
impl Default for DefaultPreprocessor {
    fn default() -> Self {
        Self {
            divide_factor: 0.25,
        }
    }
}
impl<P, V> Preprocess<P, V> for DefaultPreprocessor {
    fn divide_observations(&self, observations: &[&Observation<P, V>]) -> usize {
        let n = observations.len() as f64;
        cmp::min((self.divide_factor * n.sqrt()).ceil() as usize, 25)
    }

    fn weight_observations(
        &self,
        observations: &[&Observation<P, V>],
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
