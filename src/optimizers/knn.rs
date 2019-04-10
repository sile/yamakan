use crate::float::NonNanF64;
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::spaces::PriorDistribution;
use crate::Result;
use rand::Rng;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct KnnOptimizer<P, V>
where
    P: PriorDistribution,
{
    param_space: P,
    obss: HashMap<ObsId, Obs<P::Param, V>>,
}
impl<P, V> KnnOptimizer<P, V>
where
    P: PriorDistribution,
{
    pub fn new(param_space: P) -> Self {
        Self {
            param_space,
            obss: HashMap::new(),
        }
    }

    pub fn param_space(&self) -> &P {
        &self.param_space
    }

    pub fn param_space_mut(&mut self) -> &mut P {
        &mut self.param_space
    }
}
impl<P, V> Optimizer for KnnOptimizer<P, V>
where
    P: PriorDistribution<Param = f64>, // TODO: remove `Param=f64`
    V: Ord,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        let k = (self.obss.len() as f64).sqrt() as usize;
        let k2 = (self.obss.len() as f64).sqrt().ceil() as usize;

        let mut obss = self.obss.values().collect::<Vec<_>>();

        obss.sort_by_key(|o| &o.value);
        let superiors = obss.iter().take(k).map(|o| o.id).collect::<HashSet<_>>();

        let param = self
            .param_space
            .sample_iter(rng)
            .take(::std::cmp::max(1, k2) * 2)
            .map(|param| {
                // TODO: optimize
                let mut os = obss
                    .iter()
                    .map(|o| (o.id, NonNanF64::new((o.param - param).abs())))
                    .collect::<Vec<_>>();
                os.sort_by_key(|o| o.1);
                let c = os
                    .iter()
                    .take(k)
                    .filter(|o| superiors.contains(&o.0))
                    .count();
                (param, c)
            })
            .max_by_key(|x| x.1)
            .map(|x| x.0)
            .unwrap_or_else(|| unreachable!());
        track!(Obs::new(idg, param))
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        self.obss.insert(obs.id, obs);
        Ok(())
    }

    fn forget(&mut self, id: ObsId) -> Result<()> {
        self.obss.remove(&id);
        Ok(())
    }
}
impl<P, V> Default for KnnOptimizer<P, V>
where
    P: PriorDistribution + Default,
{
    fn default() -> Self {
        Self::new(P::default())
    }
}
