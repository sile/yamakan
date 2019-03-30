use crate::budget::Budgeted;
use crate::observation::{IdGenerator, Observation};
use crate::{Optimizer, Result};
use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::marker::PhantomData;
use std::num::NonZeroUsize;

#[derive(Debug)]
pub struct AshaOptions {
    pub r: NonZeroUsize,
    pub s: NonZeroUsize,
    pub eta: usize,
}
impl Default for AshaOptions {
    fn default() -> Self {
        Self {
            r: unsafe { NonZeroUsize::new_unchecked(1) },
            s: unsafe { NonZeroUsize::new_unchecked(4) },
            eta: 0,
        }
    }
}

#[derive(Debug)]
struct Rungs<P, V> {
    rungs: Vec<Rung<P, V>>,
}
impl<P, V> Rungs<P, V> {
    fn new() -> Self {
        Self { rungs: Vec::new() }
    }
}

#[derive(Debug)]
struct Rung<P, V> {
    observations: Vec<Observation<Option<Budgeted<P>>, V>>,
}

#[derive(Debug)]
pub struct AshaOptimizer<O: Optimizer, V> {
    inner: O,
    options: AshaOptions,
    rungs: Rungs<O::Param, O::Value>,
    _value: PhantomData<V>,
}
impl<O, V> AshaOptimizer<O, V>
where
    O: Optimizer<Value = RungValue<V>>,
{
    pub fn new(inner: O) -> Self {
        Self::with_options(inner, AshaOptions::default())
    }

    pub fn with_options(inner: O, options: AshaOptions) -> Self {
        Self {
            inner,
            options,
            rungs: Rungs::new(),
            _value: PhantomData,
        }
    }

    pub fn inner_ref(&self) -> &O {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }
}
impl<O, V> Optimizer for AshaOptimizer<O, V>
where
    O: Optimizer<Value = RungValue<V>>,
{
    type Param = Budgeted<O::Param>;
    type Value = V;

    fn ask<R: Rng, G: IdGenerator>(
        &mut self,
        rng: &mut R,
        idgen: &mut G,
    ) -> Result<Observation<Self::Param, ()>> {
        panic!()
    }

    fn tell(&mut self, observation: Observation<Self::Param, Self::Value>) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RungValue<T> {
    pub rung_num: usize,
    pub value: T,
}
impl<T: PartialOrd> PartialOrd for RungValue<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (Reverse(self.rung_num), &self.value).partial_cmp(&(Reverse(other.rung_num), &other.value))
    }
}
impl<T: Ord> Ord for RungValue<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (Reverse(self.rung_num), &self.value).cmp(&(Reverse(other.rung_num), &other.value))
    }
}
