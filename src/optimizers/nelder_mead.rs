#![allow(missing_docs)]
use crate::observation::{IdGen, Obs, ObsId};
use crate::parameters::Continuous;
use crate::{ErrorKind, Optimizer, Result};
use rand::Rng;
use rustats::num::FiniteF64;
use std;

/// Adaptive Nelder-Mead Simplex (ANMS) algorithms.
///
/// # References
///
/// - Fuchang Gao and Lixing Han (2010), Springer US. "Implementing the Nelder-Mead simplex algorithm with adaptive parameters" (doi:10.1007/s10589-010-9329-3)
/// - Saša Singer and John Nelder (2009), Scholarpedia, 4(7):2928. "Nelder-Mead algorithm" (doi:10.4249/scholarpedia.2928)
#[derive(Debug)]
pub struct NelderMeadOptimizer<P, V> {
    param_space: Vec<P>,
    simplex: Vec<Pair<V>>,
    alpha: FiniteF64,
    beta: FiniteF64,
    gamma: FiniteF64,
    delta: FiniteF64,
    initial: Option<Vec<FiniteF64>>,
    centroid: Vec<FiniteF64>,
    evaluating: Option<ObsId>,
    state: State<V>,
}
impl<P, V> NelderMeadOptimizer<P, V>
where
    P: Continuous,
    V: Ord,
{
    pub fn new(param_space: Vec<P>, x0: &[P::Param]) -> Result<Self> {
        track_assert!(
            x0.len() >= 2,
            ErrorKind::InvalidInput,
            "Too few dimensions: {}",
            x0.len()
        );

        let dim = x0.len() as f64;
        let x0 = param_space
            .iter()
            .zip(x0.iter())
            .map(|(p, x)| track!(p.encode(x)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            param_space,
            simplex: Vec::with_capacity(x0.len() + 1),
            alpha: track!(FiniteF64::new(1.0))?,
            beta: track!(FiniteF64::new(1.0 + 2.0 / dim))?,
            gamma: track!(FiniteF64::new(0.75 - 1.0 / (2.0 * dim)))?,
            delta: track!(FiniteF64::new(1.0 - 1.0 / dim))?,
            initial: Some(x0),
            centroid: Vec::new(),
            evaluating: None,
            state: State::Initialize,
        })
    }

    fn dim(&self) -> usize {
        self.param_space.len()
    }

    fn adjust(&self, x: Vec<FiniteF64>) -> Result<Vec<P::Param>> {
        self.param_space
            .iter()
            .zip(x.into_iter())
            .map(|(p, v)| {
                let r = p.range();
                let v = r.low.get().max(v.get());
                let mut v = (r.high.get() - std::f64::EPSILON).min(v);
                for i in 2.. {
                    if v != r.high.get() {
                        break;
                    }
                    v -= std::f64::EPSILON * i as f64;
                }
                let v = track!(FiniteF64::new(v))?;
                track!(p.decode(v))
            })
            .collect()
    }

    fn initial_ask(&mut self) -> Vec<FiniteF64> {
        if let Some(x0) = self.initial.take() {
            x0
        } else {
            let i = self.simplex.len() - 1;
            let tau = if self.simplex[0].param[i].get() == 0.0 {
                0.00025
            } else {
                0.05
            };
            let x = self.simplex[0]
                .param
                .iter()
                .map(|x0| x0.get())
                .enumerate()
                .map(|(j, x0)| if i == j { x0 + tau } else { x0 })
                .map(|x| unsafe { FiniteF64::new_unchecked(x) })
                .collect();
            x
        }
    }

    fn initial_tell(&mut self, obs: Obs<Vec<FiniteF64>, V>) {
        let pair = Pair {
            param: obs.param,
            value: obs.value,
        };
        self.simplex.push(pair);

        if self.simplex.len() == self.dim() + 1 {
            self.simplex.sort_by(|a, b| a.value.cmp(&b.value));
            self.update_centroid();
            self.state = State::Reflect;
        }
    }

    fn reflect_ask(&mut self) -> Vec<FiniteF64> {
        self.centroid
            .iter()
            .zip(self.xh().iter())
            .map(|(&x0, &xh)| x0 + self.alpha * (x0 - xh))
            .collect()
    }

    fn reflect_tell(&mut self, obs: Obs<Vec<FiniteF64>, V>) {
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

    fn expand_ask(&mut self, prev: Vec<FiniteF64>) -> Vec<FiniteF64> {
        self.centroid
            .iter()
            .zip(prev.iter())
            .map(|(&c, &x)| c + self.beta * (x - c))
            .collect()
    }

    fn expand_tell(&mut self, prev: Obs<Vec<FiniteF64>, V>, curr: Obs<Vec<FiniteF64>, V>) {
        if prev.value < curr.value {
            self.accept(prev);
        } else {
            self.accept(curr);
        }
    }

    fn contract_outside_ask(&mut self, prev: Vec<FiniteF64>) -> Vec<FiniteF64> {
        self.centroid
            .iter()
            .zip(prev.iter())
            .map(|(&c, &x)| c + self.gamma * (x - c))
            .collect()
    }

    fn contract_outside_tell(
        &mut self,
        prev: Obs<Vec<FiniteF64>, V>,
        curr: Obs<Vec<FiniteF64>, V>,
    ) {
        if curr.value <= prev.value {
            self.accept(curr);
        } else {
            self.shrink();
        }
    }

    fn contract_inside_ask(&mut self, prev: Vec<FiniteF64>) -> Vec<FiniteF64> {
        self.centroid
            .iter()
            .zip(prev.iter())
            .map(|(&c, &x)| c - self.gamma * (x - c))
            .collect()
    }

    fn contract_inside_tell(
        &mut self,
        _prev: Obs<Vec<FiniteF64>, V>,
        curr: Obs<Vec<FiniteF64>, V>,
    ) {
        if curr.value < self.highest().value {
            self.accept(curr);
        } else {
            self.shrink();
        }
    }

    fn shrink_ask(&mut self, index: usize) -> Vec<FiniteF64> {
        self.lowest()
            .param
            .iter()
            .zip(self.simplex[index].param.iter())
            .map(|(&xl, &xi)| xl + self.delta * (xi - xl))
            .collect()
    }

    fn shrink_tell(&mut self, obs: Obs<Vec<FiniteF64>, V>, index: usize) {
        self.simplex[index] = Pair {
            param: obs.param,
            value: obs.value,
        };
        if index < self.simplex.len() - 1 {
            self.state = State::Shrink { index: index + 1 };
        } else {
            self.update_centroid();
            self.state = State::Reflect;
        }
    }

    fn accept(&mut self, obs: Obs<Vec<FiniteF64>, V>) {
        // TODO: optimize
        self.simplex.push(Pair {
            param: obs.param,
            value: obs.value,
        });
        self.simplex.sort_by(|a, b| a.value.cmp(&b.value));
        self.simplex.pop();
        self.update_centroid();
        self.state = State::Reflect;
    }

    fn shrink(&mut self) {
        self.state = State::Shrink { index: 1 };
    }

    fn lowest(&self) -> &Pair<V> {
        &self.simplex[0]
    }

    fn second_highest(&self) -> &Pair<V> {
        &self.simplex[self.simplex.len() - 2]
    }

    fn highest(&self) -> &Pair<V> {
        &self.simplex[self.simplex.len() - 1]
    }

    // TODO: delete
    fn xh(&self) -> &[FiniteF64] {
        &self.simplex.last().unwrap_or_else(|| unreachable!()).param
    }

    fn update_centroid(&mut self) {
        assert!(self.simplex.len() == self.dim() + 1);

        // NOTE: We assume that `self.simplex` have been sorted by its values.

        let n = self.dim();
        let mut c = vec![FiniteF64::default(); n];
        for t in self.simplex.iter().take(n) {
            for i in 0..n {
                c[i] += t.param[i];
            }
        }

        let n = unsafe { FiniteF64::new_unchecked(n as f64) };
        for c in &mut c {
            *c /= n;
        }
        self.centroid = c
    }
}
impl<P, V> Optimizer for NelderMeadOptimizer<P, V>
where
    P: Continuous,
    V: Ord,
{
    type Param = Vec<P::Param>;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, _rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param>> {
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

        let x = track!(self.adjust(x))?;
        let obs = track!(Obs::new(idg, x))?;
        self.evaluating = Some(obs.id);

        Ok(obs)
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        track_assert_eq!(self.evaluating, Some(obs.id), ErrorKind::UnknownObservation);
        self.evaluating = None;

        let obs = track!(obs.try_map_param(|xs| self
            .param_space
            .iter()
            .zip(xs.iter())
            .map(|(p, x)| track!(p.encode(x)))
            .collect::<Result<Vec<_>>>()))?;

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

    fn forget(&mut self, _id: ObsId) -> Result<()> {
        Ok(())
    }
}

// TODO:replace with Obs?
#[derive(Debug, Clone)]
struct Pair<V> {
    param: Vec<FiniteF64>,
    value: V,
}

#[derive(Debug, Clone)]
enum State<V> {
    Initialize,
    Reflect,
    Expand(Obs<Vec<FiniteF64>, V>),
    ContractOutside(Obs<Vec<FiniteF64>, V>),
    ContractInside(Obs<Vec<FiniteF64>, V>),
    Shrink { index: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::SerialIdGenerator;
    use crate::parameters::F64;
    use rand;
    use trackable::result::TopLevelResult;

    fn objective(param: &[f64]) -> FiniteF64 {
        FiniteF64::new(param[0].powi(2) - param[1]).unwrap_or_else(|e| panic!("{}", e))
    }

    #[test]
    fn nelder_mead_optimizer_works() -> TopLevelResult {
        let param_space = vec![F64::new(0.0, 100.0)?, F64::new(0.0, 100.0)?];
        let mut optimizer = NelderMeadOptimizer::new(param_space, &[10.0, 20.0])?;
        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        for i in 0..100 {
            let obs = optimizer.ask(&mut rng, &mut idg)?;
            let value = objective(&obs.param);
            println!("[{}] param={:?}, value={}", i, obs.param, value.get());

            optimizer.tell(obs.map_value(|_| value))?;
        }

        Ok(())
    }
}
