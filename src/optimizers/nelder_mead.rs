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
    beta: f64,
    gamma: f64,
    delta: f64,
    initial: Option<Vec<FiniteF64>>,
    centroid: Vec<FiniteF64>,
    evaluating: Option<ObsId>,
    state: State,
}
impl<P, V> NelderMeadOptimizer<P, V>
where
    P: Continuous,
    V: Ord,
{
    pub fn new(param_space: Vec<P>, x0: &[P::Param]) -> Result<Self> {
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
            beta: 1.0 + 2.0 / dim,
            gamma: 0.75 - 1.0 / (2.0 * dim),
            delta: 1.0 - 1.0 / dim,
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
                let v = (r.high.get() - std::f64::EPSILON).min(v);
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

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param>> {
        track_assert!(self.evaluating.is_none(), ErrorKind::Other);

        let x = match self.state {
            State::Initialize => self.initial_ask(),
            State::Reflect => self.reflect_ask(),
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

        match self.state {
            State::Initialize => {
                self.initial_tell(obs);
            }
            State::Reflect => panic!(),
        }

        Ok(())
    }

    fn forget(&mut self, _id: ObsId) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Pair<V> {
    param: Vec<FiniteF64>,
    value: V,
}

#[derive(Debug, Clone, Copy)]
enum State {
    Initialize,
    Reflect,
}
