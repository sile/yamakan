//! Observation and its identifier.
use crate::{IdGen, Result};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std;

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
}

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
