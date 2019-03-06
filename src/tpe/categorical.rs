use super::{DefaultTpeStrategy, Observation, TpeStrategy};
use crate::float::NonNanF64;
use crate::optimizer::Optimizer;
use crate::space::CategoricalSpace;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::Rng;
use std::marker::PhantomData;

pub struct TpeCategoricalOptimizerBuilder<P, V, S = DefaultTpeStrategy> {
    strategy: S,
    prior_weight: f64,
    _param_space: PhantomData<P>,
    _value: PhantomData<V>,
}
impl<P, V, S> TpeCategoricalOptimizerBuilder<P, V, S>
where
    P: CategoricalSpace,
    V: Ord,
    S: TpeStrategy<P::Param, V> + Default,
{
    pub fn new() -> Self {
        Self {
            strategy: S::default(),
            prior_weight: 1.0,
            _param_space: PhantomData,
            _value: PhantomData,
        }
    }
}
impl<P, V, S> TpeCategoricalOptimizerBuilder<P, V, S>
where
    P: CategoricalSpace,
    V: Ord,
    S: TpeStrategy<P::Param, V>,
{
    pub fn strategy<S1>(self, strategy: S1) -> TpeCategoricalOptimizerBuilder<P, V, S1> {
        TpeCategoricalOptimizerBuilder {
            strategy,
            prior_weight: self.prior_weight,
            _param_space: PhantomData,
            _value: PhantomData,
        }
    }

    pub fn prior_weight(mut self, weight: f64) -> Self {
        self.prior_weight = weight;
        self
    }

    pub fn finish(self, param_space: P) -> TpeCategoricalOptimizer<P, V, S> {
        TpeCategoricalOptimizer {
            observations: Vec::new(),
            strategy: self.strategy,
            prior_weight: self.prior_weight,
            param_space,
        }
    }
}

#[derive(Debug)]
pub struct TpeCategoricalOptimizer<P: CategoricalSpace, V, S = DefaultTpeStrategy> {
    observations: Vec<Observation<P::Param, V>>,
    strategy: S,
    param_space: P,
    prior_weight: f64,
}
impl<P, V, S> TpeCategoricalOptimizer<P, V, S>
where
    P: CategoricalSpace,
    V: Ord,
    S: TpeStrategy<P::Param, V> + Default,
{
    pub fn new(param_space: P) -> Self {
        TpeCategoricalOptimizerBuilder::new().finish(param_space)
    }
}
impl<P, V, S> Optimizer for TpeCategoricalOptimizer<P, V, S>
where
    P: CategoricalSpace,
    V: Ord,
    S: TpeStrategy<P::Param, V>,
{
    type Param = P::Param;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let (superiors, inferiors) = self.strategy.divide_observations(&self.observations);
        let superior_weights = self.strategy.weight_superiors(superiors);
        let inferior_weights = self.strategy.weight_superiors(inferiors);
        assert_eq!(superiors.len(), superior_weights.len());
        assert_eq!(inferiors.len(), inferior_weights.len());

        let superior_histogram = Histogram::new(
            superiors
                .iter()
                .map(|o| &o.param)
                .zip(superior_weights.into_iter()),
            &self.param_space,
            self.prior_weight,
        );
        let inferior_histogram = Histogram::new(
            inferiors
                .iter()
                .map(|o| &o.param)
                .zip(inferior_weights.into_iter()),
            &self.param_space,
            self.prior_weight,
        );

        let mut indices = (0..self.param_space.size().get()).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices
            .into_iter()
            .map(|i| self.param_space.index_to_param(i))
            .map(|category| {
                let superior_log_likelihood = superior_histogram.pdf(&category).ln();
                let inferior_log_likelihood = inferior_histogram.pdf(&category).ln();
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
impl<'a, P: CategoricalSpace> Histogram<'a, P> {
    fn new<I>(observations: I, param_space: &'a P, prior_weight: f64) -> Self
    where
        I: Iterator<Item = (&'a P::Param, f64)>,
    {
        let mut probabilities = vec![prior_weight; param_space.size().get()];
        for (param, weight) in observations {
            probabilities[param_space.param_to_index(param)] += weight;
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

    // TODO: s/pdf/probability/
    fn pdf(&self, category: &P::Param) -> f64 {
        self.probabilities[self.param_space.param_to_index(category)]
    }
}
impl<'a, P: CategoricalSpace> Distribution<P::Param> for Histogram<'a, P> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> P::Param {
        self.param_space.index_to_param(self.dist.sample(rng))
    }
}
