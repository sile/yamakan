//! A collection of Black-Box Optimization algorithms.
//!
//! "yamakan" is a Japanese translation of "guesswork".
#[macro_use]
extern crate trackable;

pub use self::error::{Error, ErrorKind};

pub mod budget;
pub mod observation;
pub mod optimizers;
pub mod parameters;

mod error;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, Error>;
