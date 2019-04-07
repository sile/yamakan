use crate::budget::{Budget, Budgeted};
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::Result;
use rand::Rng;
use std::cmp::{self, Ordering, Reverse};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

#[derive(Debug)]
pub struct AshaOptions {
    pub r: NonZeroUsize,
    pub s: usize,
    pub eta: NonZeroUsize,
    pub max_suspended: NonZeroUsize,
}
impl Default for AshaOptions {
    fn default() -> Self {
        Self {
            r: unsafe { NonZeroUsize::new_unchecked(1) },
            s: 0,
            eta: unsafe { NonZeroUsize::new_unchecked(4) },
            max_suspended: unsafe { NonZeroUsize::new_unchecked(16) },
        }
    }
}

#[derive(Debug)]
pub struct AshaOptimizer<O: Optimizer, V> {
    inner: O,
    max_budget: u64,
    options: AshaOptions,
    rungs: Rungs<O::Param, V>,
    total_suspended: usize,
    _value: PhantomData<V>,
}
impl<O, V> AshaOptimizer<O, V>
where
    O: Optimizer<Value = RungValue<V>>,
    V: Clone + Ord,
{
    pub fn new(inner: O, max_budget: u64) -> Self {
        Self::with_options(inner, max_budget, AshaOptions::default())
    }

    pub fn with_options(inner: O, max_budget: u64, options: AshaOptions) -> Self {
        Self {
            inner,
            max_budget,
            options,
            rungs: Rungs::new(),
            total_suspended: 0,
            _value: PhantomData,
        }
    }

    pub fn inner_ref(&self) -> &O {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    pub fn options(&self) -> &AshaOptions {
        &self.options
    }

    fn get_rung_num(&self, c: u64) -> usize {
        let AshaOptions { r, s, eta, .. } = self.options;
        let n = c / r.get() as u64;
        if n == 0 {
            return 0;
        }

        let rung = (n as f64).log(eta.get() as f64) as usize;
        if rung <= s {
            0
        } else {
            rung - s
        }
    }

    fn abandone_one(&mut self) {
        for rung in &mut self.rungs.rungs {
            let worst = rung
                .observations
                .iter()
                .filter(|x| x.1.is_suspended())
                .max_by_key(|x| x.1.value());
            if let Some((id, entry)) = worst {
                let entry = RungEntry::Finished {
                    value: entry.value().clone(),
                };
                rung.observations.insert(*id, entry);
                return;
            }
        }
        unreachable!()
    }
}
impl<O, V> Optimizer for AshaOptimizer<O, V>
where
    O: Optimizer<Value = RungValue<V>>,
    O::Param: Clone,
    V: Clone + Ord,
{
    type Param = Budgeted<O::Param>;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        let mut rung_num = self.rungs.rungs.len();

        for rung in self.rungs.rungs.iter_mut().rev() {
            rung_num -= 1;

            let best_suspended = rung
                .observations
                .iter()
                .filter(|x| x.1.is_suspended())
                .min_by_key(|x| x.1.value());
            if let Some((&id, RungEntry::Suspended { param, value })) = best_suspended {
                let superiors = rung
                    .observations
                    .values()
                    .filter(|v| !v.is_suspended())
                    .filter(|v| v.value() < value)
                    .count();
                if superiors >= rung.observations.len() / self.options.eta.get() {
                    continue;
                }

                let new_budget = self.options.r.get()
                    * self
                        .options
                        .eta
                        .get()
                        .pow((rung_num + 1) as u32 + self.options.s as u32);
                let mut param = param.clone();
                param
                    .budget_mut()
                    .set_amount(cmp::min(self.max_budget, new_budget as u64));

                let entry = RungEntry::Finished {
                    value: value.clone(),
                };
                rung.observations.insert(id, entry);

                let obs = Obs {
                    id,
                    param,
                    value: (),
                };
                return Ok(obs);
            }
        }

        let new_obs = track!(self.inner.ask(rng, idg))?;
        Ok(new_obs.map_param(|p| {
            let amount = self.options.r.get() * self.options.eta.get().pow(self.options.s as u32);
            let budget = Budget::new(cmp::min(self.max_budget, amount as u64));
            Budgeted::new(budget, p)
        }))
    }

    fn tell(&mut self, observation: Obs<Self::Param, Self::Value>) -> Result<()> {
        let rung_num = self.get_rung_num(observation.param.budget().consumption());
        let rung = self.rungs.get_mut(rung_num);
        if observation.param.budget().consumption() >= self.max_budget {
            let inner_observation = Obs {
                id: observation.id,
                param: observation.param.get().clone(),
                value: RungValue {
                    rung_num: rung_num + 1,
                    value: observation.value,
                },
            };
            track!(self.inner.tell(inner_observation))?;
        } else if let Some(entry) = rung.observations.get_mut(&observation.id) {
            if let RungEntry::Suspended { param, .. } = entry {
                *param = observation.param.clone();
            }
        } else {
            let entry = RungEntry::Suspended {
                param: observation.param.clone(),
                value: observation.value.clone(),
            };
            rung.observations.insert(observation.id, entry);

            let inner_observation = Obs {
                id: observation.id,
                param: observation.param.get().clone(),
                value: RungValue {
                    rung_num,
                    value: observation.value,
                },
            };
            self.total_suspended += 1;

            track!(self.inner.tell(inner_observation))?;
        }

        while self.options.max_suspended.get() < self.total_suspended {
            self.abandone_one();
            self.total_suspended -= 1;
        }

        Ok(())
    }

    fn forget(&mut self, _id: ObsId) -> Result<()> {
        unimplemented!()
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

    fn get_mut(&mut self, i: usize) -> &mut Rung<P, V> {
        assert!(i < 1000, "i={}", i); // TODO
        for _ in self.rungs.len()..=i {
            self.rungs.push(Rung::new());
        }
        self.rungs
            .get_mut(i)
            .unwrap_or_else(|| panic!("never fails"))
    }
}

#[derive(Debug)]
enum RungEntry<P, V> {
    Suspended { param: Budgeted<P>, value: V },
    Finished { value: V },
}
impl<P, V> RungEntry<P, V> {
    fn is_suspended(&self) -> bool {
        if let RungEntry::Suspended { .. } = self {
            true
        } else {
            false
        }
    }

    fn value(&self) -> &V {
        match self {
            RungEntry::Suspended { value, .. } | RungEntry::Finished { value } => value,
        }
    }
}

#[derive(Debug)]
struct Rung<P, V> {
    observations: HashMap<ObsId, RungEntry<P, V>>,
}
impl<P, V> Rung<P, V> {
    fn new() -> Self {
        Self {
            observations: HashMap::new(),
        }
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

// TODO: #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::observation::SerialIdGenerator;
//     use crate::optimizers::random::RandomOptimizer;
//     use crate::spaces::F64;
//     use rand;
//     use trackable::result::TestResult;

//     #[test]
//     fn get_rung_num_works() {
//         let inner = RandomOptimizer::new(F64 {
//             low: 0.0,
//             high: 1.0,
//         });
//         let optimizer = AshaOptimizer::<_, usize>::new(inner, 64);

//         assert_eq!(optimizer.get_rung_num(0), 0);
//         assert_eq!(optimizer.get_rung_num(1), 0);
//         assert_eq!(optimizer.get_rung_num(3), 0);
//         assert_eq!(optimizer.get_rung_num(4), 1);
//         assert_eq!(optimizer.get_rung_num(15), 1);
//         assert_eq!(optimizer.get_rung_num(16), 2);
//     }

//     #[test]
//     fn asha_works() -> TestResult {
//         let inner = RandomOptimizer::new(F64 {
//             low: 0.0,
//             high: 1.0,
//         });
//         let mut optimizer = AshaOptimizer::<_, usize>::new(inner, 64);

//         let mut rng = rand::thread_rng();
//         let mut idg = SerialIdGenerator::new();
//         let obs = track!(optimizer.ask(&mut rng, &mut idg))?;
//         let obs = obs.map_value(|_| 1);
//         track!(optimizer.tell(obs))?;

//         Ok(())
//     }
// }
