use super::{DefaultPreprocessor, Preprocess, TpeOptions};
use crate::float::NonNanF64;
use crate::observation::{IdGen, Obs, ObsId};
use crate::optimizers::Optimizer;
use crate::spaces::{Categorical, PriorPmf};
use crate::Result;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

/// TPE Optimizer for categorical parameter.
#[derive(Debug)]
pub struct TpeCategoricalOptimizer<P: Categorical, V, T = DefaultPreprocessor> {
    param_space: P,
    options: TpeOptions<T>,
    observations: HashMap<ObsId, Obs<P::Param, V>>,
}
impl<P, V, T> TpeCategoricalOptimizer<P, V, T>
where
    P: Categorical + PriorPmf,
    V: Ord,
    T: Preprocess<P::Param, V> + Default,
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
    T: Preprocess<P::Param, V>,
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
    T: Preprocess<P::Param, V>,
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

        let superior_histogram = Histogram::new(
            superiors.iter().map(|o| &o.param).zip(superior_weights),
            &self.param_space,
            self.options.prior_weight,
        );
        let inferior_histogram = Histogram::new(
            inferiors.iter().map(|o| &o.param).zip(inferior_weights),
            &self.param_space,
            self.options.prior_weight,
        );

        let mut indices = (0..self.param_space.size()).collect::<Vec<_>>();
        indices.shuffle(rng); // for tie break
        let param = indices
            .into_iter()
            .map(|i| self.param_space.from_index(i))
            .map(|param| {
                let superior_log_likelihood = superior_histogram.pmf(&param).ln();
                let inferior_log_likelihood = inferior_histogram.pmf(&param).ln();
                let ei = superior_log_likelihood - inferior_log_likelihood;
                (ei, param)
            })
            .max_by_key(|(ei, _)| NonNanF64::new(*ei))
            .map(|(_, param)| param)
            .unwrap_or_else(|| unreachable!());
        track!(Obs::new(idg, param))
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
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
    dist: WeightedIndex<f64>,
    param_space: &'a P,
}
impl<'a, P> Histogram<'a, P>
where
    P: Categorical + PriorPmf,
{
    fn new<I>(observations: I, param_space: &'a P, prior_weight: f64) -> Self
    where
        I: Iterator<Item = (&'a P::Param, f64)>,
    {
        let mut probabilities = (0..param_space.size())
            .map(|i| {
                let p = param_space.from_index(i);
                param_space.pmf(&p) * prior_weight
            })
            .collect::<Vec<_>>();
        for (param, weight) in observations {
            probabilities[param_space.to_index(param)] += weight;
        }

        let sum = probabilities.iter().sum::<f64>();
        for p in &mut probabilities {
            *p /= sum;
        }

        let dist =
            WeightedIndex::new(probabilities.iter()).unwrap_or_else(|e| unreachable!("{}", e));
        Self {
            probabilities,
            dist,
            param_space,
        }
    }

    fn pmf(&self, param: &P::Param) -> f64 {
        self.probabilities[self.param_space.to_index(param)]
    }
}
impl<'a, P> Distribution<P::Param> for Histogram<'a, P>
where
    P: Categorical,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> P::Param {
        self.param_space.from_index(self.dist.sample(rng))
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
