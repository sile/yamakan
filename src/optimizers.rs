//! Black-box optimizers.
use crate::observation::{IdGen, Obs, ObsId};
use crate::Result;
use rand::Rng;

pub mod asha;
pub mod hyperband;
pub mod random;

/// Black-box optimizer.
pub trait Optimizer {
    /// The parameter to be optimized.
    type Param;

    /// The value obtained as a result of a parameter evaluation.
    type Value;

    /// Asks the next parameter to be evaluated.
    ///
    /// The evaluation result should be told to this optimizer.
    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param>>;

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

    /// Forgets the observation associated with the given ID.
    ///
    /// # Errors
    ///
    /// Some implementations may return an `ErrorKind::UnknownObservation` error
    /// if this optimizer does not known (or has not generated) the specified observation.
    fn forget(&mut self, id: ObsId) -> Result<()>;
}
