use crate::Result;

#[derive(Debug)]
pub struct Observation<P, V> {
    pub id: ObservationId,
    pub param: P,
    pub value: V,
}
impl<P> Observation<P, ()> {
    pub fn new<G: IdGenerator>(idgen: &mut G, param: P) -> Result<Self> {
        let id = track!(idgen.generate())?;
        Ok(Observation {
            id,
            param,
            value: (),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationId(u64);
impl ObservationId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// Observation ID generator.
pub trait IdGenerator {
    fn generate(&mut self) -> Result<ObservationId>;
}
