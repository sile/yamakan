#[macro_use]
extern crate trackable;

pub use self::error::{Error, ErrorKind};

pub mod budget;
pub mod observation;
pub mod optimizers;
pub mod spaces;

mod error;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, Error>;
