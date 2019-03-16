use super::{DefaultPreprocessor, Preprocess};
use crate::float::NonNanF64;
use crate::optimizer::{Observation, Optimizer};
use crate::space::ParamSpace;
use failure::Error;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug)]
pub struct TpeCategoricalOptions<T> {
    preprocessor: T,
    prior_weight: f64,
}
impl<T> TpeCategoricalOptions<T> {
    pub fn new(preprocessor: T) -> Self {
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
impl<T: Default> Default for TpeCategoricalOptions<T> {
    fn default() -> Self {
        TpeCategoricalOptions {
            preprocessor: T::default(),
            prior_weight: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct TpeCategoricalOptimizer<P, V, T = DefaultPreprocessor>
where
    P: ParamSpace<Internal = usize>,
{
    param_space: P,
    options: TpeCategoricalOptions<T>,
    observations: Vec<Observation<P::External, V>>,
}
impl<P, V, T> TpeCategoricalOptimizer<P, V, T>
where
    P: ParamSpace<Internal = usize>,
    V: Ord,
    T: Preprocess<P::External, V> + Default,
{
    pub fn new(param_space: P) -> Self {
        Self::with_options(param_space, TpeCategoricalOptions::default())
    }
}
impl<P, V, T> TpeCategoricalOptimizer<P, V, T>
where
    P: ParamSpace<Internal = usize>,
    V: Ord,
    T: Preprocess<P::External, V>,
{
    pub fn with_options(param_space: P, options: TpeCategoricalOptions<T>) -> Self {
        Self {
            param_space,
            options,
            observations: Vec::new(),
        }
    }

    pub fn param_space(&self) -> &P {
        &self.param_space
    }
}
impl<P, V, T> Optimizer for TpeCategoricalOptimizer<P, V, T>
where
    P: ParamSpace<Internal = usize>,
    V: Ord,
    T: Preprocess<P::External, V>,
{
    type Param = P::External;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        // FIXME: optimize (buffer new observations in `tell` and merge them with existing ones)
        self.observations.sort_by(|a, b| a.value.cmp(&b.value));

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
            .map(|i| self.param_space.externalize(i))
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
        self.observations.push(o);
    }
}

#[derive(Debug)]
struct Histogram<'a, P> {
    probabilities: Vec<f64>,
    dist: WeightedIndex<f64>,
    param_space: &'a P,
}
impl<'a, P: ParamSpace<Internal = usize>> Histogram<'a, P> {
    fn new<I>(observations: I, param_space: &'a P, prior_weight: f64) -> Self
    where
        I: Iterator<Item = (&'a P::External, f64)>,
    {
        let low = param_space.internal_range().start;
        let space_size = param_space.internal_range().end - low;
        let mut probabilities = vec![prior_weight; space_size];
        for (param, weight) in observations {
            probabilities[param_space.internalize(param) - low] += weight;
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

    fn pmf(&self, category: &P::External) -> f64 {
        let low = self.param_space.internal_range().start;
        self.probabilities[self.param_space.internalize(category) - low]
    }
}
impl<'a, P: ParamSpace<Internal = usize>> Distribution<P::External> for Histogram<'a, P> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> P::External {
        let low = self.param_space.internal_range().start;
        self.param_space.externalize(&(self.dist.sample(rng) + low))
    }
}
