use super::{DefaultPreprocessor, Preprocess, TpeOptions};
use crate::float::NonNanF64;
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::spaces::{Categorical, PriorPmf};
use crate::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

/// TPE optimizer for categorical parameter.
#[derive(Debug)]
pub struct TpeCategoricalOptimizer<P, V, T = DefaultPreprocessor> {
    param_space: P,
    options: TpeOptions<T>,
    observations: HashMap<ObsId, Obs<usize, V>>,
}
impl<P, V, T> TpeCategoricalOptimizer<P, V, T>
where
    P: Categorical + PriorPmf,
    V: Ord,
    T: Preprocess<V> + Default,
{
    /// Makes a new `TpeCategoricalOptimizer` instance.
    pub fn new(param_space: P) -> Self {
        Self::with_options(param_space, TpeOptions::default())
    }
}
impl<P, V, T> TpeCategoricalOptimizer<P, V, T>
where
    P: Categorical + PriorPmf,
    V: Ord,
    T: Preprocess<V>,
{
    /// Makes a new `TpeCategoricalOptimizer` instance with the given options.
    pub fn with_options(param_space: P, options: TpeOptions<T>) -> Self {
        Self {
            param_space,
            options,
            observations: HashMap::new(),
        }
    }

    /// Returns a reference to the parameter space.
    pub fn param_space(&self) -> &P {
        &self.param_space
    }

    /// Returns a mutable reference to the parameter space.
    pub fn param_space_mut(&mut self) -> &mut P {
        &mut self.param_space
    }
}
impl<P, V, T> Optimizer for TpeCategoricalOptimizer<P, V, T>
where
    P: Categorical + PriorPmf,
    V: Ord,
    T: Preprocess<V>,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: &mut R, idg: &mut G) -> Result<Obs<Self::Param, ()>> {
        let mut observations = self.observations.values().collect::<Vec<_>>();
        observations.sort_by_key(|o| &o.value);

        let gamma = self.options.preprocessor.divide_observations(&observations);
        let (superiors, inferiors) = observations.split_at(gamma);

        let superior_weights = self
            .options
            .preprocessor
            .weight_observations(superiors, true);
        let inferior_weights = self
            .options
            .preprocessor
            .weight_observations(inferiors, false);

        let superior_histogram = track!(Histogram::new(
            superiors.iter().map(|o| o.param).zip(superior_weights),
            &self.param_space,
            self.options.prior_weight,
        ))?;
        let inferior_histogram = track!(Histogram::new(
            inferiors.iter().map(|o| o.param).zip(inferior_weights),
            &self.param_space,
            self.options.prior_weight,
        ))?;

        let mut indices = (0..self.param_space.size()).collect::<Vec<_>>();
        indices.shuffle(rng); // for tie break
        let (_, param) = indices
            .into_iter()
            .map(|candidate| {
                let superior_log_likelihood = superior_histogram.pmf(candidate).ln();
                let inferior_log_likelihood = inferior_histogram.pmf(candidate).ln();
                let ei = superior_log_likelihood - inferior_log_likelihood;
                (ei, candidate)
            })
            .max_by_key(|(ei, _)| NonNanF64::new(*ei))
            .unwrap_or_else(|| unreachable!());
        let param = track!(self.param_space.from_index(param))?;
        track!(Obs::new(idg, param))
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        let obs = track!(obs.try_map_param(|p| self.param_space.to_index(&p)))?;
        self.observations.insert(obs.id, obs);
        Ok(())
    }

    fn forget(&mut self, id: ObsId) -> Result<()> {
        self.observations.remove(&id);
        Ok(())
    }
}

#[derive(Debug)]
struct Histogram<'a, P> {
    probabilities: Vec<f64>,
    param_space: &'a P,
}
impl<'a, P> Histogram<'a, P>
where
    P: Categorical + PriorPmf,
{
    fn new<I>(observations: I, param_space: &'a P, prior_weight: f64) -> Result<Self>
    where
        I: Iterator<Item = (usize, f64)>,
    {
        let mut probabilities = (0..param_space.size())
            .map(|i| {
                let p = track!(param_space.from_index(i); i)?;
                Ok(param_space.pmf(&p) * prior_weight)
            })
            .collect::<Result<Vec<_>>>()?;
        for (param, weight) in observations {
            probabilities[param] += weight;
        }

        let sum = probabilities.iter().sum::<f64>();
        for p in &mut probabilities {
            *p /= sum;
        }

        Ok(Self {
            probabilities,
            param_space,
        })
    }

    fn pmf(&self, param: usize) -> f64 {
        self.probabilities[param]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::SerialIdGenerator;
    use crate::spaces::Bool;
    use rand;
    use trackable::result::TestResult;

    #[test]
    fn tpe_categorical_works() -> TestResult {
        let mut opt = TpeCategoricalOptimizer::<_, usize>::new(Bool);
        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.tell(obs.map_value(|_| 10)))?;

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.forget(obs.id))?;

        Ok(())
    }
}
