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
    simplex: Vec<Vec<FiniteF64>>,
    values: Vec<V>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    evaluating: Option<ObsId>,
}
impl<P, V> NelderMeadOptimizer<P, V>
where
    P: Continuous,
{
    pub fn new(param_space: Vec<P>, x0: &[P::Param]) -> Result<Self> {
        let dim = x0.len() as f64;
        let x0 = param_space
            .iter()
            .zip(x0.iter())
            .map(|(p, x)| track!(p.encode(x)))
            .collect::<Result<_>>()?;
        Ok(Self {
            param_space,
            simplex: vec![x0],
            values: vec![],
            alpha: 1.0,
            beta: 1.0 + 2.0 / dim,
            gamma: 0.75 - 1.0 / (2.0 * dim),
            delta: 1.0 - 1.0 / dim,
            evaluating: None,
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

    fn initial_ask(&self) -> Result<Vec<FiniteF64>> {
        let x = track_assert_some!(self.simplex.last(), ErrorKind::Bug);
        Ok(x.clone())
    }

    fn initial_tell<T>(&mut self, obs: Obs<Vec<T>, V>) -> Result<()> {
        self.values.push(obs.value);

        if self.simplex.len() <= self.dim() {
            let i = self.simplex.len() - 1;
            let tau = if self.simplex[0][i].get() == 0.0 {
                0.00025
            } else {
                0.05
            };
            let x = self.simplex[0]
                .iter()
                .map(|x0| x0.get())
                .enumerate()
                .map(|(j, x0)| if i == j { x0 + tau } else { x0 })
                .map(|x| unsafe { FiniteF64::new_unchecked(x) })
                .collect();
            self.simplex.push(x);
        }
        Ok(())
    }
}
impl<P, V> Optimizer for NelderMeadOptimizer<P, V>
where
    P: Continuous,
{
    type Param = Vec<P::Param>;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param>> {
        track_assert!(self.evaluating.is_none(), ErrorKind::Other);

        let x = if self.values.len() <= self.dim() {
            track!(self.initial_ask())?
        } else {
            panic!()
        };

        let x = track!(self.adjust(x))?;
        let obs = track!(Obs::new(idg, x))?;
        self.evaluating = Some(obs.id);

        Ok(obs)
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        track_assert_eq!(self.evaluating, Some(obs.id), ErrorKind::UnknownObservation);
        self.evaluating = None;

        if self.values.len() <= self.dim() {
            track!(self.initial_tell(obs))?;
        } else {
            panic!()
        }

        Ok(())
    }

    fn forget(&mut self, _id: ObsId) -> Result<()> {
        Ok(())
    }
}
