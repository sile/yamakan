//! Observation and its identifier.
use crate::{Budget, IdGen, Result};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std;

/// Observation Identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ObsId(u64);
impl ObsId {
    /// Makes a new observation identifier.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the value of this identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Observation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Obs<P, V = ()> {
    /// Observation identifier.
    pub id: ObsId,

    /// Evaluation parameter.
    pub param: P,

    /// Observed value.
    pub value: V,
}
impl<P> Obs<P, ()> {
    /// Makes a new unevaluated observation.
    pub fn new<G: IdGen>(mut idg: G, param: P) -> Result<Self> {
        let id = track!(idg.generate())?;
        Ok(Self {
            id,
            param,
            value: (),
        })
    }
}
impl<P, V> Obs<P, V> {
    /// Updates the parameter by the result of the given function.
    pub fn map_param<F, Q>(self, f: F) -> Obs<Q, V>
    where
        F: FnOnce(P) -> Q,
    {
        Obs {
            id: self.id,
            param: f(self.param),
            value: self.value,
        }
    }

    /// Tries updating the parameter by the result of the given function.
    pub fn try_map_param<F, Q, E>(self, f: F) -> std::result::Result<Obs<Q, V>, E>
    where
        F: FnOnce(P) -> std::result::Result<Q, E>,
    {
        Ok(Obs {
            id: self.id,
            param: f(self.param)?,
            value: self.value,
        })
    }

    /// Updates the value by the result of the given function.
    pub fn map_value<F, U>(self, f: F) -> Obs<P, U>
    where
        F: FnOnce(V) -> U,
    {
        Obs {
            id: self.id,
            param: self.param,
            value: f(self.value),
        }
    }

    /// Tries updating the value by the result of the given function.
    pub fn try_map_value<F, U, E>(self, f: F) -> std::result::Result<Obs<P, U>, E>
    where
        F: FnOnce(V) -> std::result::Result<U, E>,
    {
        Ok(Obs {
            id: self.id,
            param: self.param,
            value: f(self.value)?,
        })
    }

    /// Takes the value of this observation.
    pub fn take_value(self) -> (Obs<P>, V) {
        let Obs { id, param, value } = self;
        (
            Obs {
                id,
                param,
                value: (),
            },
            value,
        )
    }
}
impl<P, V> From<MfObs<P, V>> for Obs<P, V> {
    fn from(f: MfObs<P, V>) -> Self {
        Self {
            id: f.id,
            param: f.param,
            value: f.value,
        }
    }
}

/// Multi-Fidelity Observation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MfObs<P, V = ()> {
    /// Observation identifier.
    pub id: ObsId,

    /// Evaluation budget.
    pub budget: Budget,

    /// Evaluation parameter.
    pub param: P,

    /// Observed value.
    pub value: V,
}
impl<P> MfObs<P, ()> {
    /// Makes a new unevaluated observation.
    pub fn new<G: IdGen>(mut idg: G, budget: Budget, param: P) -> Result<Self> {
        let id = track!(idg.generate())?;
        Ok(Self {
            id,
            budget,
            param,
            value: (),
        })
    }
}
impl<P, V> MfObs<P, V> {
    /// Makes a `MfObs` instance from an observation and budget.
    pub fn from_obs(obs: Obs<P, V>, budget: Budget) -> Self {
        Self {
            id: obs.id,
            budget,
            param: obs.param,
            value: obs.value,
        }
    }
}
impl<P, V> MfObs<P, V> {
    /// Updates the parameter by the result of the given function.
    pub fn map_param<F, Q>(self, f: F) -> MfObs<Q, V>
    where
        F: FnOnce(P) -> Q,
    {
        MfObs {
            id: self.id,
            budget: self.budget,
            param: f(self.param),
            value: self.value,
        }
    }

    /// Tries updating the parameter by the result of the given function.
    pub fn try_map_param<F, Q, E>(self, f: F) -> std::result::Result<MfObs<Q, V>, E>
    where
        F: FnOnce(P) -> std::result::Result<Q, E>,
    {
        Ok(MfObs {
            id: self.id,
            budget: self.budget,
            param: f(self.param)?,
            value: self.value,
        })
    }

    /// Updates the value by the result of the given function.
    pub fn map_value<F, U>(self, f: F) -> MfObs<P, U>
    where
        F: FnOnce(V) -> U,
    {
        MfObs {
            id: self.id,
            budget: self.budget,
            param: self.param,
            value: f(self.value),
        }
    }

    /// Tries updating the value by the result of the given function.
    pub fn try_map_value<F, U, E>(self, f: F) -> std::result::Result<MfObs<P, U>, E>
    where
        F: FnOnce(V) -> std::result::Result<U, E>,
    {
        Ok(MfObs {
            id: self.id,
            budget: self.budget,
            param: self.param,
            value: f(self.value)?,
        })
    }

    /// Takes the value of this observation.
    pub fn take_value(self) -> (MfObs<P>, V) {
        let MfObs {
            id,
            budget,
            param,
            value,
        } = self;
        (
            MfObs {
                id,
                budget,
                param,
                value: (),
            },
            value,
        )
    }
}
