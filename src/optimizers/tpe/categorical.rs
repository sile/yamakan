use crate::float::NonNanF64;
use crate::optimizer::{Observation, Optimizer};
use crate::optimizers::tpe::{DefaultPreprocessor, Preprocess};
use crate::space::SearchSpace;
use failure::Error;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug)]
pub struct TpeCategoricalOptions<P> {
    preprocessor: P,
    prior_weight: f64,
}
impl<P> TpeCategoricalOptions<P> {
    pub fn new(preprocessor: P) -> Self {
        Self {
            preprocessor,
            prior_weight: 1.0,
        }
    }

    pub fn prior_weight(mut self, weight: f64) -> Result<Self, Error> {
        ensure!(weight > 0.0, "weight={}", weight);
        ensure!(weight.is_finite(), "weight={}", weight);

        self.prior_weight = weight;
        Ok(self)
    }
}
impl<P: Default> Default for TpeCategoricalOptions<P> {
    fn default() -> Self {
        TpeCategoricalOptions {
            preprocessor: P::default(),
            prior_weight: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct TpeCategoricalOptimizer<S, V, P = DefaultPreprocessor>
where
    S: SearchSpace<InternalParam = usize>,
{
    param_space: S,
    options: TpeCategoricalOptions<P>,
    observations: Vec<Observation<S::ExternalParam, V>>,
}
impl<S, V, P> TpeCategoricalOptimizer<S, V, P>
where
    S: SearchSpace<InternalParam = usize>,
    V: Ord,
    P: Preprocess<S::ExternalParam, V> + Default,
{
    pub fn new(param_space: S) -> Self {
        Self::with_options(param_space, TpeCategoricalOptions::default())
    }
}
impl<S, V, P> TpeCategoricalOptimizer<S, V, P>
where
    S: SearchSpace<InternalParam = usize>,
    V: Ord,
    P: Preprocess<S::ExternalParam, V>,
{
    pub fn with_options(param_space: S, options: TpeCategoricalOptions<P>) -> Self {
        Self {
            param_space,
            options,
            observations: Vec::new(),
        }
    }
}
impl<S, V, P> Optimizer for TpeCategoricalOptimizer<S, V, P>
where
    S: SearchSpace<InternalParam = usize>,
    V: Ord,
    P: Preprocess<S::ExternalParam, V>,
{
    type Param = S::ExternalParam;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let gamma = self
            .options
            .preprocessor
            .divide_observations(&self.observations);
        let (superiors, inferiors) = self.observations.split_at(gamma);
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

        let space_size =
            self.param_space.internal_range().end - self.param_space.internal_range().start;
        let mut indices = (0..space_size).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices
            .iter()
            .map(|i| self.param_space.to_external(i))
            .map(|category| {
                let superior_log_likelihood = superior_histogram.pmf(&category).ln();
                let inferior_log_likelihood = inferior_histogram.pmf(&category).ln();
                let ei = superior_log_likelihood - inferior_log_likelihood;
                (ei, category)
            })
            .max_by_key(|(ei, _)| NonNanF64::new(*ei))
            .map(|(_, category)| category)
            .expect("never fails")
    }

    fn tell(&mut self, param: Self::Param, value: Self::Value) {
        let o = Observation { param, value };
        let i = self
            .observations
            .binary_search_by(|x| x.value.cmp(&o.value))
            .unwrap_or_else(|i| i);
        self.observations.insert(i, o);
    }
}

#[derive(Debug)]
struct Histogram<'a, P> {
    probabilities: Vec<f64>,
    dist: WeightedIndex<f64>,
    param_space: &'a P,
}
impl<'a, P: SearchSpace<InternalParam = usize>> Histogram<'a, P> {
    fn new<I>(observations: I, param_space: &'a P, prior_weight: f64) -> Self
    where
        I: Iterator<Item = (&'a P::ExternalParam, f64)>,
    {
        let space_size = param_space.internal_range().end - param_space.internal_range().start;
        let mut probabilities = vec![prior_weight; space_size];
        for (param, weight) in observations {
            probabilities[param_space.to_internal(param)] += weight;
        }

        let sum = probabilities.iter().sum::<f64>();
        for p in &mut probabilities {
            *p /= sum;
        }

        let dist = WeightedIndex::new(probabilities.iter()).expect("never fails");
        Self {
            probabilities,
            dist,
            param_space,
        }
    }

    fn pmf(&self, category: &P::ExternalParam) -> f64 {
        self.probabilities[self.param_space.to_internal(category)]
    }
}
impl<'a, P: SearchSpace<InternalParam = usize>> Distribution<P::ExternalParam>
    for Histogram<'a, P>
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> P::ExternalParam {
        self.param_space.to_external(&self.dist.sample(rng))
    }
}
