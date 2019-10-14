//! Observation identifier generators.
use crate::{IdGen, ObsId, Result};

/// An implementation of `IdGen` that generates serial identifiers starting from zero.
#[derive(Debug, Default)]
pub struct SerialIdGenerator {
    next_id: u64,
}
impl SerialIdGenerator {
    /// Makes a new `SerialIdGenerator` instance.
    pub const fn new() -> Self {
        Self { next_id: 0 }
    }
}
impl IdGen for SerialIdGenerator {
    fn generate(&mut self) -> Result<ObsId> {
        let id = self.next_id;
        self.next_id += 1;
        Ok(ObsId::new(id))
    }
}

/// An implementation of `IdGen` that always returns the same identifier.
#[derive(Debug)]
pub struct ConstIdGenerator {
    id: ObsId,
}
impl ConstIdGenerator {
    /// Makes a new `ConstIdGenerator` instance.
    ///
    /// When `ConstIdGenerator::generate` method is called, it always returns the given identifier.
    pub const fn new(id: ObsId) -> Self {
        Self { id }
    }
}
impl IdGen for ConstIdGenerator {
    fn generate(&mut self) -> Result<ObsId> {
        Ok(self.id)
    }
}
