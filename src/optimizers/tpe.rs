//! TPE(**T**ree-structured **P**arzen **E**stimator) optimizer.
//!
//! # References
//!
//! - [Algorithms for Hyper-Parameter Optimization][TPE]
//!
//! [TPE]: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
pub use self::categorical::TpeCategoricalOptimizer;
pub use self::numerical::TpeNumericalOptimizer;
pub use self::options::TpeOptions;
pub use self::preprocess::{DefaultPreprocessor, Preprocess};

mod categorical;
mod numerical;
mod options;
mod parzen_estimator;
mod preprocess;
