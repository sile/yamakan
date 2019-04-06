use crate::Result;

/// Observation.
#[derive(Debug)]
pub struct Obs<P, V = ()> {
    pub id: ObsId,
    pub param: P,
    pub value: V,
}
impl<P> Obs<P, ()> {
    pub fn new<G: IdGen>(idg: &mut G, param: P) -> Result<Self> {
        let id = track!(idg.generate())?;
        Ok(Self {
            id,
            param,
            value: (),
        })
    }
}
impl<P, V> Obs<P, V> {
    pub fn map_param<F, Q>(self, f: F) -> Obs<Q, V>
    where
        F: FnOnce(P) -> Q,
    {
        Obs {
            id: self.id,
            param: f(self.param),
            value: self.value,
        }
    }

    pub fn map_value<F, U>(self, f: F) -> Obs<P, U>
    where
        F: FnOnce(V) -> U,
    {
        Obs {
            id: self.id,
            param: self.param,
            value: f(self.value),
        }
    }
}

/// Observation Identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObsId(u64);
impl ObsId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// Observation ID generator.
pub trait IdGen {
    fn generate(&mut self) -> Result<ObsId>;
}

#[derive(Debug, Default)]
pub struct SerialIdGenerator {
    next_id: u64,
}
impl SerialIdGenerator {
    pub fn new() -> Self {
        Self::default()
    }
}
impl IdGen for SerialIdGenerator {
    fn generate(&mut self) -> Result<ObsId> {
        let id = self.next_id;
        self.next_id += 1;
        Ok(ObsId::new(id))
    }
}
