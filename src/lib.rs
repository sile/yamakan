#[macro_use]
extern crate trackable;

pub use self::error::{Error, ErrorKind};
pub use self::space::ParamSpace; // TODO: delete

pub mod budget;
pub mod observation;
pub mod optimizers;
pub mod range;
pub mod spaces;

mod error;
mod float;
mod iter;
mod space;

/// This crate specific `Result` type.
pub type Result<T> = std::result::Result<T, Error>;
