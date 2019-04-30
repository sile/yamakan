#[macro_use]
extern crate trackable;

pub use self::error::{Error, ErrorKind};

pub mod budget;
pub mod observation;
pub mod optimizers;
pub mod range;
pub mod spaces;

mod error;
mod iter;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, Error>;
