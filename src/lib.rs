#[macro_use]
extern crate trackable;

pub use self::error::{Error, ErrorKind};
pub use self::optimizer::{Observation, Optimizer};
pub use self::space::ParamSpace;

pub mod budget;
pub mod optimizers;
pub mod spaces;

mod error;
mod float;
mod iter;
mod optimizer;
mod space;

pub type Result<T> = std::result::Result<T, Error>;
