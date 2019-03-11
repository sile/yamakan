use super::{DefaultTpeStrategy, TpeStrategy};
use crate::float::NonNanF64;
use crate::optimizer::{Observation, Optimizer};
use crate::space::SearchSpace;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::Rng;
use std::marker::PhantomData;

// TODO(?): s/P/S/
pub struct TpeCategoricalOptimizerBuilder<P, V, S = DefaultTpeStrategy> {
    strategy: S,
    prior_weight: f64,
    _param_space: PhantomData<P>,
    _value: PhantomData<V>,
}
impl<P, V, S> TpeCategoricalOptimizerBuilder<P, V, S>
where
    P: SearchSpace<InternalParam = usize>,
    V: Ord,
    S: TpeStrategy<P::ExternalParam, V> + Default,
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
    P: SearchSpace<InternalParam = usize>,
    V: Ord,
    S: TpeStrategy<P::ExternalParam, V>,
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
pub struct TpeCategoricalOptimizer<P: SearchSpace<InternalParam = usize>, V, S = DefaultTpeStrategy>
{
    observations: Vec<Observation<P::ExternalParam, V>>,
    strategy: S,
    param_space: P,
    prior_weight: f64,
}
impl<P, V, S> TpeCategoricalOptimizer<P, V, S>
where
    P: SearchSpace<InternalParam = usize>,
    V: Ord,
    S: TpeStrategy<P::ExternalParam, V> + Default,
{
    pub fn new(param_space: P) -> Self {
        TpeCategoricalOptimizerBuilder::new().finish(param_space)
    }
}
impl<P, V, S> Optimizer for TpeCategoricalOptimizer<P, V, S>
where
    P: SearchSpace<InternalParam = usize>,
    V: Ord,
    S: TpeStrategy<P::ExternalParam, V>,
{
    type Param = P::ExternalParam;
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

        let space_size =
            self.param_space.internal_range().end - self.param_space.internal_range().start;
        let mut indices = (0..space_size).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices
            .iter()
            .map(|i| self.param_space.to_external(i))
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

    // TODO: s/pdf/probability/
    fn pdf(&self, category: &P::ExternalParam) -> f64 {
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
