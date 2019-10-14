//! A collection of Black-Box Optimization algorithms.
//!
//! "yamakan" is a Japanese translation of "guesswork".
#![warn(missing_docs)]

#[macro_use]
extern crate trackable;

use rand::Rng;

pub use self::error::{Error, ErrorKind};
pub use self::observation::{Obs, ObsId};

pub mod budget;
pub mod domains;
pub mod generators;
pub mod optimizers;

mod error;
mod observation;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, Error>;

/// This trait provides ask-and-tell interface for black-box optimization.
pub trait Optimizer {
    /// The parameter to be optimized.
    type Param;

    /// The value obtained as a result of a parameter evaluation.
    type Value;

    /// Asks the next parameter to be evaluated.
    ///
    /// The evaluation result should be told to this optimizer.
    fn ask<R: Rng, G: IdGen>(&mut self, rng: R, idg: G) -> Result<Obs<Self::Param>>;

    /// Tells the result of an observation to this optimizer.
    ///
    /// If there is an existing observation that has the same identifier,
    /// the state of the observation should be overwritten by the new one.
    ///
    /// # Errors
    ///
    /// Some implementations may return an `ErrorKind::UnknownObservation` error
    /// if this optimizer does not known (or has not generated) the specified observation.
    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()>;
}

/// This trait provides ask-and-tell interface for multi-fidelity black-box optimization.
pub trait MultiFidelityOptimizer {
    /// The parameter to be optimized.
    type Param;

    /// The value obtained as a result of a parameter evaluation.
    type Value;

    /// Asks the next parameter to be evaluated.
    ///
    /// The evaluation result should be told to this optimizer.
    fn ask<R: Rng, G: IdGen>(&mut self, rng: R, idg: G) -> Result<Obs<Self::Param>>;

    /// Tells the result of an observation to this optimizer.
    ///
    /// If there is an existing observation that has the same identifier,
    /// the state of the observation should be overwritten by the new one.
    ///
    /// # Errors
    ///
    /// Some implementations may return an `ErrorKind::UnknownObservation` error
    /// if this optimizer does not known (or has not generated) the specified observation.
    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<bool>;
}

/// Parameter search domain.
pub trait Domain {
    /// A specific point in this domain.
    type Point;
}

/// Observation ID generator.
pub trait IdGen {
    /// Generates a new identifier.
    fn generate(&mut self) -> Result<ObsId>;
}
impl<'a, T: IdGen + ?Sized> IdGen for &'a mut T {
    fn generate(&mut self) -> Result<ObsId> {
        (**self).generate()
    }
}
impl<T: IdGen + ?Sized> IdGen for Box<T> {
    fn generate(&mut self) -> Result<ObsId> {
        (**self).generate()
    }
}
