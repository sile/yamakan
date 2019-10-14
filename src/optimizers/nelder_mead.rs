//! Adaptive nelder-mead simplex algorithm.
//!
//! # References
//!
//! - [Implementing the Nelder-Mead simplex algorithm with adaptive parameters][ANMS]
//! - [Nelder-Mead algorithm](http://var.scholarpedia.org/article/Nelder-Mead_algorithm)
//! - [Nelder-Mead Method (Wikipedia)](https://en.wikipedia.org/wiki/Nelder–Mead_method)
//!
//! [ANMS]: https://link.springer.com/article/10.1007/s10589-010-9329-3
use crate::domains::ContinuousDomain;
use crate::{ErrorKind, IdGen, Obs, ObsId, Optimizer, Result};
use rand::distributions::Distribution;
use rand::Rng;
use std;
use std::f64::EPSILON;

/// An optimizer based on [Adaptive Nelder-Mead Simplex (ANMS)][ANMS] algorithm.
///
/// [ANMS]: https://link.springer.com/article/10.1007/s10589-010-9329-3
#[derive(Debug)]
pub struct NelderMeadOptimizer<V> {
    params_domain: Vec<ContinuousDomain>,
    simplex: Vec<Obs<Vec<f64>, V>>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    initial: Vec<Vec<f64>>,
    centroid: Vec<f64>,
    evaluating: Option<ObsId>,
    state: State<V>,
}
impl<V> NelderMeadOptimizer<V>
where
    V: Ord,
{
    /// Makes a new `NelderMeadOptimizer`.
    pub fn new<R: Rng>(params_domain: Vec<ContinuousDomain>, mut rng: R) -> Result<Self> {
        let point = params_domain
            .iter()
            .map(|p| p.sample(&mut rng))
            .collect::<Vec<_>>();
        track!(Self::with_initial_point(params_domain, &point))
    }

    /// Makes a new `NelderMeadOptimizer` which has the given search point.
    pub fn with_initial_point(params_domain: Vec<ContinuousDomain>, point: &[f64]) -> Result<Self> {
        let mut initial_simplex = vec![point.to_vec()];
        for i in 0..params_domain.len() {
            let tau = if point[i] == 0.0 { 0.00025 } else { 0.05 };
            let x = point
                .iter()
                .enumerate()
                .map(|(j, &x0)| if i == j { x0 + tau } else { x0 })
                .collect();
            initial_simplex.push(x);
        }
        track!(Self::with_initial_simplex(params_domain, initial_simplex))
    }

    /// Makes a new `NelderMeadOptimizer` with the given simplex.
    pub fn with_initial_simplex(
        params_domain: Vec<ContinuousDomain>,
        initial_simplex: Vec<Vec<f64>>,
    ) -> Result<Self> {
        track_assert!(
            params_domain.len() >= 2,
            ErrorKind::InvalidInput,
            "Too few dimensions: {}",
            params_domain.len()
        );
        track_assert_eq!(
            params_domain.len() + 1,
            initial_simplex.len(),
            ErrorKind::InvalidInput
        );

        let dim = params_domain.len() as f64;
        Ok(Self {
            params_domain,
            simplex: Vec::with_capacity(initial_simplex.len()),
            alpha: 1.0,
            beta: 1.0 + 2.0 / dim,
            gamma: 0.75 - 1.0 / (2.0 * dim),
            delta: 1.0 - 1.0 / dim,
            initial: initial_simplex,
            centroid: Vec::new(),
            evaluating: None,
            state: State::Initialize,
        })
    }

    fn dim(&self) -> usize {
        self.params_domain.len()
    }

    fn adjust(&self, x: Vec<f64>) -> Vec<f64> {
        self.params_domain
            .iter()
            .zip(x.into_iter())
            .map(|(p, v)| {
                let v = p.low().max(v);
                let mut v = (p.high() - std::f64::EPSILON).min(v);
                for i in 2.. {
                    if (v - p.high()).abs() > EPSILON {
                        break;
                    }
                    v -= EPSILON * f64::from(i);
                }
                v
            })
            .collect()
    }

    fn initial_ask(&mut self) -> Vec<f64> {
        self.initial.pop().unwrap_or_else(|| unreachable!())
    }

    fn initial_tell(&mut self, obs: Obs<Vec<f64>, V>) {
        self.simplex.push(obs);

        if self.simplex.len() == self.dim() + 1 {
            self.simplex.sort_by(|a, b| a.value.cmp(&b.value));
            self.update_centroid();
            self.state = State::Reflect;
        }
    }

    fn reflect_ask(&mut self) -> Vec<f64> {
        self.centroid
            .iter()
            .zip(self.highest().param.iter())
            .map(|(&x0, &xh)| x0 + self.alpha * (x0 - xh))
            .collect()
    }

    fn reflect_tell(&mut self, obs: Obs<Vec<f64>, V>) {
        if obs.value < self.lowest().value {
            self.state = State::Expand(obs);
        } else if obs.value < self.second_highest().value {
            self.accept(obs);
        } else if obs.value < self.highest().value {
            self.state = State::ContractOutside(obs);
        } else {
            self.state = State::ContractInside(obs);
        }
    }

    fn expand_ask(&mut self, prev: Vec<f64>) -> Vec<f64> {
        self.centroid
            .iter()
            .zip(prev.iter())
            .map(|(&c, &x)| c + self.beta * (x - c))
            .collect()
    }

    fn expand_tell(&mut self, prev: Obs<Vec<f64>, V>, curr: Obs<Vec<f64>, V>) {
        if prev.value < curr.value {
            self.accept(prev);
        } else {
            self.accept(curr);
        }
    }

    fn contract_outside_ask(&mut self, prev: Vec<f64>) -> Vec<f64> {
        self.centroid
            .iter()
            .zip(prev.iter())
            .map(|(&c, &x)| c + self.gamma * (x - c))
            .collect()
    }

    fn contract_outside_tell(&mut self, prev: Obs<Vec<f64>, V>, curr: Obs<Vec<f64>, V>) {
        if curr.value <= prev.value {
            self.accept(curr);
        } else {
            self.shrink();
        }
    }

    fn contract_inside_ask(&mut self, prev: Vec<f64>) -> Vec<f64> {
        self.centroid
            .iter()
            .zip(prev.iter())
            .map(|(&c, &x)| c - self.gamma * (x - c))
            .collect()
    }

    fn contract_inside_tell(&mut self, _prev: Obs<Vec<f64>, V>, curr: Obs<Vec<f64>, V>) {
        if curr.value < self.highest().value {
            self.accept(curr);
        } else {
            self.shrink();
        }
    }

    fn shrink_ask(&mut self, index: usize) -> Vec<f64> {
        self.lowest()
            .param
            .iter()
            .zip(self.simplex[index].param.iter())
            .map(|(&xl, &xi)| xl + self.delta * (xi - xl))
            .collect()
    }

    fn shrink_tell(&mut self, obs: Obs<Vec<f64>, V>, index: usize) {
        self.simplex[index] = obs;
        if index < self.simplex.len() - 1 {
            self.state = State::Shrink { index: index + 1 };
        } else {
            self.update_centroid();
            self.state = State::Reflect;
        }
    }

    fn accept(&mut self, obs: Obs<Vec<f64>, V>) {
        // FIXME: optimize
        self.simplex.push(obs);
        self.simplex.sort_by(|a, b| a.value.cmp(&b.value));
        self.simplex.pop();
        self.update_centroid();
        self.state = State::Reflect;
    }

    fn shrink(&mut self) {
        self.state = State::Shrink { index: 1 };
    }

    fn lowest(&self) -> &Obs<Vec<f64>, V> {
        &self.simplex[0]
    }

    fn second_highest(&self) -> &Obs<Vec<f64>, V> {
        &self.simplex[self.simplex.len() - 2]
    }

    fn highest(&self) -> &Obs<Vec<f64>, V> {
        &self.simplex[self.simplex.len() - 1]
    }

    fn update_centroid(&mut self) {
        assert!(self.simplex.len() == self.dim() + 1);

        // NOTE: We assume that `self.simplex` have been sorted by its values.

        let n = self.dim();
        let mut c = vec![f64::default(); n];
        for t in self.simplex.iter().take(n) {
            for (i, c) in c.iter_mut().enumerate() {
                *c += t.param[i];
            }
        }

        let n = n as f64;
        for c in &mut c {
            *c /= n;
        }
        self.centroid = c
    }
}
impl<V> Optimizer for NelderMeadOptimizer<V>
where
    V: Ord,
{
    type Param = Vec<f64>;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, _rng: R, idg: G) -> Result<Obs<Self::Param>> {
        track_assert!(self.evaluating.is_none(), ErrorKind::Other);

        let x = match &self.state {
            State::Initialize => self.initial_ask(),
            State::Reflect => self.reflect_ask(),
            State::Expand(prev) => {
                let prev = prev.param.clone();
                self.expand_ask(prev)
            }
            State::ContractOutside(prev) => {
                let prev = prev.param.clone();
                self.contract_outside_ask(prev)
            }
            State::ContractInside(prev) => {
                let prev = prev.param.clone();
                self.contract_inside_ask(prev)
            }
            State::Shrink { index } => {
                let index = *index;
                self.shrink_ask(index)
            }
        };

        let x = self.adjust(x);
        let obs = track!(Obs::new(idg, x))?;
        self.evaluating = Some(obs.id);

        Ok(obs)
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        track_assert_eq!(self.evaluating, Some(obs.id), ErrorKind::UnknownObservation);
        self.evaluating = None;

        match std::mem::replace(&mut self.state, State::Initialize) {
            State::Initialize => {
                self.initial_tell(obs);
            }
            State::Reflect => {
                self.reflect_tell(obs);
            }
            State::Expand(prev) => {
                self.expand_tell(prev, obs);
            }
            State::ContractOutside(prev) => {
                self.contract_outside_tell(prev, obs);
            }
            State::ContractInside(prev) => {
                self.contract_inside_tell(prev, obs);
            }
            State::Shrink { index } => {
                self.shrink_tell(obs, index);
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
enum State<V> {
    Initialize,
    Reflect,
    Expand(Obs<Vec<f64>, V>),
    ContractOutside(Obs<Vec<f64>, V>),
    ContractInside(Obs<Vec<f64>, V>),
    Shrink { index: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::ContinuousDomain;
    use crate::generators::SerialIdGenerator;
    use ordered_float::NotNan;
    use rand;
    use trackable::result::TopLevelResult;

    fn objective(param: &[f64]) -> f64 {
        param[0].powi(2) - param[1]
    }

    #[test]
    fn nelder_mead_optimizer_works() -> TopLevelResult {
        let params_domain = vec![
            ContinuousDomain::new(0.0, 100.0)?,
            ContinuousDomain::new(0.0, 100.0)?,
        ];
        let mut optimizer = NelderMeadOptimizer::with_initial_point(params_domain, &[10.0, 20.0])?;
        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        for i in 0..100 {
            let obs = optimizer.ask(&mut rng, &mut idg)?;
            let value = objective(&obs.param);
            println!("[{}] param={:?}, value={}", i, obs.param, value);

            optimizer
                .tell(obs.map_value(|_| NotNan::new(value).unwrap_or_else(|e| panic!("{}", e))))?;
        }

        Ok(())
    }
}
