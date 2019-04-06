use crate::budget::Budgeted;
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::asha::{AshaOptimizer, AshaOptions, RungValue};
use crate::{ErrorKind, Optimizer, Result};
use factory::Factory;
use rand::Rng;
use std::collections::HashMap;
use std::num::NonZeroUsize;

#[derive(Debug)]
pub struct HyperbandOptions {
    pub r: NonZeroUsize,
    pub eta: NonZeroUsize,
    pub max_susp: NonZeroUsize,
}
impl Default for HyperbandOptions {
    fn default() -> Self {
        Self {
            r: unsafe { NonZeroUsize::new_unchecked(1) },
            eta: unsafe { NonZeroUsize::new_unchecked(4) },
            max_susp: unsafe { NonZeroUsize::new_unchecked(4) },
        }
    }
}

pub struct HyperbandOptimizer<O: Optimizer, V> {
    brackets: Vec<Bracket<O, V>>,
    runnings: HashMap<ObsId, usize>,
}
impl<O, V> HyperbandOptimizer<O, V>
where
    O: Optimizer<Value = RungValue<V>>,
    V: Ord + Clone,
{
    pub fn new<F>(factory: F, max_budget: u64) -> Result<Self>
    where
        F: Factory<Item = Result<O>>,
    {
        track!(Self::with_options(
            factory,
            max_budget,
            HyperbandOptions::default()
        ))
    }

    pub fn with_options<F>(factory: F, max_budget: u64, options: HyperbandOptions) -> Result<Self>
    where
        F: Factory<Item = Result<O>>,
    {
        let max_bracket = (max_budget as f64).log(options.eta.get() as f64) as usize;
        let mut brackets = Vec::with_capacity(max_bracket + 1);
        for i in 0..=max_bracket {
            let asha_options = AshaOptions {
                r: options.r,
                s: i,
                eta: options.eta,
                max_suspended: options.max_susp,
            };
            let inner = track!(factory.create())?;
            let asha = AshaOptimizer::with_options(inner, max_budget, asha_options);
            brackets.push(Bracket::new(asha));
        }
        track_assert!(!brackets.is_empty(), ErrorKind::InvalidInput);

        Ok(Self {
            brackets,
            runnings: HashMap::new(),
        })
    }
}
impl<O, V> Optimizer for HyperbandOptimizer<O, V>
where
    O: Optimizer<Value = RungValue<V>>,
    O::Param: Clone,
    V: Clone + Ord,
{
    type Param = Budgeted<O::Param>;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        let (i, bracket) = track_assert_some!(
            self.brackets
                .iter_mut()
                .enumerate()
                .min_by_key(|x| x.1.consumption),
            ErrorKind::Bug
        );
        let obs = track!(bracket.asha.ask(rng, idg))?;
        bracket.consumption += obs.param.budget().remaining();

        self.runnings.insert(obs.id, i);

        Ok(obs)
    }

    fn tell(&mut self, observation: Obs<Self::Param, Self::Value>) -> Result<()> {
        let i = track_assert_some!(
            self.runnings.remove(&observation.id),
            ErrorKind::UnknownObservation
        );

        let bracket = &mut self.brackets[i];
        bracket.consumption -= observation.param.budget().remaining();
        bracket.consumption += observation.param.budget().excess();
        track!(bracket.asha.tell(observation))?;

        Ok(())
    }

    fn forget(&mut self, _id: ObsId) -> Result<()> {
        unimplemented!()
    }
}

struct Bracket<O: Optimizer, V> {
    asha: AshaOptimizer<O, V>,
    consumption: u64,
}
impl<O: Optimizer, V> Bracket<O, V> {
    fn new(asha: AshaOptimizer<O, V>) -> Self {
        Self {
            asha,
            consumption: 0,
        }
    }
}
