use super::{DefaultTpeStrategy, Observation, TpeStrategy};
use crate::float::NonNanF64;
use crate::optimizer::Optimizer;
use crate::parameter::Categorical;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::Rng;
use std::marker::PhantomData;

pub struct TpeCategoricalOptimizerBuilder<P, V, S = DefaultTpeStrategy> {
    strategy: S,
    is_paper: bool,
    _param: PhantomData<P>,
    _value: PhantomData<V>,
}
impl<P, V, S> TpeCategoricalOptimizerBuilder<P, V, S>
where
    P: Categorical,
    V: Ord,
    S: TpeStrategy<P, V> + Default,
{
    pub fn new() -> Self {
        Self {
            strategy: S::default(),
            is_paper: true,
            _param: PhantomData,
            _value: PhantomData,
        }
    }
}
impl<P, V, S> TpeCategoricalOptimizerBuilder<P, V, S>
where
    P: Categorical,
    V: Ord,
    S: TpeStrategy<P, V>,
{
    pub fn strategy<S1>(self, strategy: S1) -> TpeCategoricalOptimizerBuilder<P, V, S1> {
        TpeCategoricalOptimizerBuilder {
            strategy,
            is_paper: self.is_paper,
            _param: PhantomData,
            _value: PhantomData,
        }
    }

    pub fn is_paper(mut self, b: bool) -> Self {
        self.is_paper = b;
        self
    }

    pub fn finish(self) -> TpeCategoricalOptimizer<P, V, S> {
        TpeCategoricalOptimizer {
            observations: Vec::new(),
            strategy: self.strategy,
            is_paper: self.is_paper,
            _param: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct TpeCategoricalOptimizer<P, V, S = DefaultTpeStrategy> {
    observations: Vec<Observation<P, V>>,
    strategy: S,
    is_paper: bool,
    _param: PhantomData<P>,
}
impl<P, V, S> TpeCategoricalOptimizer<P, V, S>
where
    P: Categorical,
    S: TpeStrategy<P, V> + Default,
{
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            strategy: S::default(),
            is_paper: true,
            _param: PhantomData,
        }
    }
}
impl<P, V, S> Optimizer for TpeCategoricalOptimizer<P, V, S>
where
    P: Categorical,
    V: Ord,
    S: TpeStrategy<P, V>,
{
    type Param = P;
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
        );
        let inferior_histogram = Histogram::new(
            inferiors
                .iter()
                .map(|o| &o.param)
                .zip(inferior_weights.into_iter()),
        );

        let mut indices = (0..P::SIZE.get()).collect::<Vec<_>>();
        indices.shuffle(rng);
        indices
            .into_iter()
            .map(P::from_index)
            .map(|category| {
                if self.is_paper {
                    let superior_log_likelihood = superior_histogram.pdf(&category);
                    let inferior_log_likelihood = inferior_histogram.pdf(&category);
                    let ei = superior_log_likelihood / inferior_log_likelihood;
                    (ei, category)
                } else {
                    let superior_log_likelihood = superior_histogram.pdf(&category).ln();
                    let inferior_log_likelihood = inferior_histogram.pdf(&category).ln();
                    let ei = superior_log_likelihood - inferior_log_likelihood;
                    (ei, category)
                }
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
struct Histogram<P> {
    probabilities: Vec<f64>,
    dist: WeightedIndex<f64>,
    _param: PhantomData<P>,
}
impl<P: Categorical> Histogram<P> {
    fn new<'a, I>(observations: I) -> Self
    where
        P: 'a,
        I: Iterator<Item = (&'a P, f64)>,
    {
        let mut probabilities = vec![1.0; P::SIZE.get()];
        for (param, weight) in observations {
            probabilities[param.to_index()] += weight;
        }

        let sum = probabilities.iter().sum::<f64>();
        for p in &mut probabilities {
            *p /= sum;
        }

        let dist = WeightedIndex::new(probabilities.iter()).expect("never fails");
        Self {
            probabilities,
            dist,
            _param: PhantomData,
        }
    }

    fn pdf(&self, category: &P) -> f64 {
        self.probabilities[category.to_index()]
    }
}
impl<P: Categorical> Distribution<P> for Histogram<P> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> P {
        P::from_index(self.dist.sample(rng))
    }
}
