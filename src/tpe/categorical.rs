use super::{DefaultTpeStrategy, Observation, TpeStrategy};
use crate::float::NonNanF64;
use crate::optimizer::Optimizer;
use crate::parameter::Categorical;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

const DEFAULT_EI_CANDIDATES: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(24) };

pub struct TpeCategoricalOptimizerBuilder<P, V, S = DefaultTpeStrategy, R = ThreadRng> {
    strategy: S,
    rng: R,
    ei_candidates: NonZeroUsize,
    _param: PhantomData<P>,
    _value: PhantomData<V>,
}
impl<P, V, S, R> TpeCategoricalOptimizerBuilder<P, V, S, R>
where
    P: Categorical,
    V: Ord,
    S: TpeStrategy<P, V> + Default,
    R: Rng + Default,
{
    pub fn new() -> Self {
        Self {
            strategy: S::default(),
            rng: R::default(),
            ei_candidates: DEFAULT_EI_CANDIDATES,
            _param: PhantomData,
            _value: PhantomData,
        }
    }
}
impl<P, V, S, R> TpeCategoricalOptimizerBuilder<P, V, S, R>
where
    P: Categorical,
    V: Ord,
    S: TpeStrategy<P, V>,
    R: Rng,
{
    pub fn strategy<S1>(self, strategy: S1) -> TpeCategoricalOptimizerBuilder<P, V, S1, R> {
        TpeCategoricalOptimizerBuilder {
            strategy,
            rng: self.rng,
            ei_candidates: self.ei_candidates,
            _param: PhantomData,
            _value: PhantomData,
        }
    }

    pub fn rng<R1>(self, rng: R1) -> TpeCategoricalOptimizerBuilder<P, V, S, R1> {
        TpeCategoricalOptimizerBuilder {
            strategy: self.strategy,
            rng,
            ei_candidates: self.ei_candidates,
            _param: PhantomData,
            _value: PhantomData,
        }
    }

    pub fn ei_candidates(mut self, n: NonZeroUsize) -> Self {
        self.ei_candidates = n;
        self
    }

    pub fn finish(self) -> TpeCategoricalOptimizer<P, V, S, R> {
        TpeCategoricalOptimizer {
            observations: Vec::new(),
            strategy: self.strategy,
            rng: self.rng,
            ei_candidates: self.ei_candidates,
            _param: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct TpeCategoricalOptimizer<P, V, S = DefaultTpeStrategy, R = ThreadRng> {
    observations: Vec<Observation<P, V>>,
    strategy: S,
    rng: R,
    ei_candidates: NonZeroUsize,
    _param: PhantomData<P>,
}
impl<P, V, S, R> TpeCategoricalOptimizer<P, V, S, R>
where
    P: Categorical,
    S: TpeStrategy<P, V> + Default,
    R: Rng + Default,
{
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            strategy: S::default(),
            rng: R::default(),
            ei_candidates: DEFAULT_EI_CANDIDATES,
            _param: PhantomData,
        }
    }
}
impl<P, V, S, R> Optimizer for TpeCategoricalOptimizer<P, V, S, R>
where
    P: Categorical,
    V: Ord,
    S: TpeStrategy<P, V>,
    R: Rng,
{
    type Param = P;
    type Value = V;

    fn ask(&mut self) -> Self::Param {
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

        superior_histogram
            .sample_iter(&mut self.rng)
            .take(self.ei_candidates.get())
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
