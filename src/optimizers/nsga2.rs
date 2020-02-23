//! NSGA-II (Non-dominated Sorting Genetic Algorithm II).
//!
//! # References
//!
//! - [A fast and elitist multiobjective genetic algorithm: NSGA-II][NSGA-II]
//!
//! [NSGA-II]: https://ieeexplore.ieee.org/document/996017
use crate::domains::VecDomain;
use crate::{Domain, ErrorKind, IdGen, Obs, Optimizer, Result};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::INFINITY;
use std::marker::PhantomData;

/// This trait allows generating new individuals.
pub trait Generate<D: Domain> {
    /// Generates a new individual.
    fn generate<R: Rng>(&mut self, rng: R, domain: &D) -> Result<D::Point>;
}

/// Random generator.
#[derive(Debug, Default)]
pub struct RandomGenerator;

impl<D> Generate<D> for RandomGenerator
where
    D: Domain + Distribution<<D as Domain>::Point>,
{
    fn generate<R: Rng>(&mut self, mut rng: R, domain: &D) -> Result<D::Point> {
        Ok(domain.sample(&mut rng))
    }
}

/// This trait allows selecting parents from a population.
pub trait Select<D: Domain> {
    /// Select a parent.
    fn select<'a, R: Rng>(
        &mut self,
        rng: R,
        population: &'a [Obs<D::Point, Vec<f64>>],
    ) -> Result<&'a Obs<D::Point, Vec<f64>>>;
}

/// Tournament selector.
#[derive(Debug)]
pub struct TournamentSelector {
    tournament_size: usize,
}

impl TournamentSelector {
    /// Makes a new `TournamentSelector` instance.
    ///
    /// # Error
    ///
    /// If `tournament_size` is less than `2`, this returns an `ErrorKind::InvalidInput` error.
    pub fn new(tournament_size: usize) -> Result<Self> {
        track_assert!(tournament_size > 1, ErrorKind::InvalidInput; tournament_size);
        Ok(Self { tournament_size })
    }
}

impl Default for TournamentSelector {
    fn default() -> Self {
        Self { tournament_size: 2 }
    }
}

impl<D: Domain> Select<D> for TournamentSelector {
    fn select<'a, R: Rng>(
        &mut self,
        mut rng: R,
        population: &'a [Obs<D::Point, Vec<f64>>],
    ) -> Result<&'a Obs<D::Point, Vec<f64>>> {
        let mut winner = track_assert_some!(population.choose(&mut rng), ErrorKind::InvalidInput);
        for _ in 1..self.tournament_size {
            let candidate =
                track_assert_some!(population.choose(&mut rng), ErrorKind::InvalidInput);
            if track!(dominates(candidate, winner))? {
                winner = candidate;
            }
        }
        Ok(winner)
    }
}

/// This trait allows applying crossover operator.
pub trait CrossOver<D: Domain> {
    /// Applies crossover operator.
    fn cross_over<R: Rng>(&mut self, rng: R, p0: &mut D::Point, p1: &mut D::Point) -> Result<()>;
}

/// This trait allows applying mutation operator.
pub trait Mutate<D: Domain> {
    /// Mutates an individual.
    fn mutate<R: Rng>(&mut self, rng: R, domain: &D, p: &mut D::Point) -> Result<()>;
}

/// A crossover operator that stochastically exchanges two individuals.
#[derive(Debug)]
pub struct Exchange {
    probability: f64,
}

impl Exchange {
    /// Makes a new `Exchange` instance.
    pub fn new(probability: f64) -> Result<Self> {
        track_assert!(0.0 <= probability && probability <= 1.0, ErrorKind::InvalidInput; probability);
        Ok(Self { probability })
    }
}

impl Default for Exchange {
    fn default() -> Self {
        Self { probability: 0.5 }
    }
}

impl<D: Domain> CrossOver<D> for Exchange {
    fn cross_over<R: Rng>(
        &mut self,
        mut rng: R,
        p0: &mut D::Point,
        p1: &mut D::Point,
    ) -> Result<()> {
        if rng.gen_bool(self.probability) {
            std::mem::swap(p0, p1);
        }
        Ok(())
    }
}

/// Vector version of `Exchange` operator.
#[derive(Debug, Default)]
pub struct ExchangeVec(Exchange);

impl ExchangeVec {
    /// Makes a new `ExchangeVec` instance.
    pub fn new(probability: f64) -> Result<Self> {
        track!(Exchange::new(probability)).map(Self)
    }
}

impl<D: Domain> CrossOver<VecDomain<D>> for ExchangeVec
where
    Exchange: CrossOver<D>,
{
    fn cross_over<R: Rng>(
        &mut self,
        mut rng: R,
        ps0: &mut Vec<D::Point>,
        ps1: &mut Vec<D::Point>,
    ) -> Result<()> {
        track_assert_eq!(ps0.len(), ps1.len(), ErrorKind::InvalidInput);
        for (p0, p1) in ps0.iter_mut().zip(ps1.iter_mut()) {
            track!(self.0.cross_over(&mut rng, p0, p1))?;
        }
        Ok(())
    }
}

/// A mutation operator that stochastically replaces a individual with a randomly sampled value.
#[derive(Debug)]
pub struct Replace {
    probability: f64,
}

impl Replace {
    /// Makes a new `Replace` instance.
    pub fn new(probability: f64) -> Result<Self> {
        track_assert!(0.0 <= probability && probability <= 1.0, ErrorKind::InvalidInput; probability);
        Ok(Self { probability })
    }
}

impl Default for Replace {
    fn default() -> Self {
        Self { probability: 0.3 }
    }
}

impl<D> Mutate<D> for Replace
where
    D: Domain + Distribution<<D as Domain>::Point>,
{
    fn mutate<R: Rng>(&mut self, mut rng: R, domain: &D, p: &mut D::Point) -> Result<()> {
        if rng.gen_bool(self.probability) {
            *p = domain.sample(&mut rng);
        }
        Ok(())
    }
}

/// Vector version of `Replace` operator.
#[derive(Debug, Default)]
pub struct ReplaceVec(Replace);

impl ReplaceVec {
    /// Makes a new `ReplaceVec` instance.
    pub fn new(probability: f64) -> Result<Self> {
        track!(Replace::new(probability)).map(Self)
    }
}

impl<D> Mutate<VecDomain<D>> for ReplaceVec
where
    D: Domain + Distribution<<D as Domain>::Point>,
    Replace: Mutate<D>,
{
    fn mutate<R: Rng>(
        &mut self,
        mut rng: R,
        domain: &VecDomain<D>,
        ps: &mut Vec<D::Point>,
    ) -> Result<()> {
        for (d, p) in domain.0.iter().zip(ps.iter_mut()) {
            track!(self.0.mutate(&mut rng, d, p))?;
        }
        Ok(())
    }
}

fn dominates<P>(a: &Obs<P, Vec<f64>>, b: &Obs<P, Vec<f64>>) -> Result<bool> {
    track_assert_eq!(a.value.len(), b.value.len(), ErrorKind::InvalidInput);
    if a.value.iter().zip(b.value.iter()).any(|(a, b)| a > b) {
        Ok(false)
    } else {
        Ok(a.value.iter().zip(b.value.iter()).any(|(a, b)| a < b))
    }
}

/// This trait allows providing operators used by the NSGA-II algorithm.
pub trait Strategy<D: Domain> {
    /// Generator.
    type Generator: Generate<D>;

    /// Selector.
    type Selector: Select<D>;

    /// Crossover.
    type CrossOver: CrossOver<D>;

    /// Mutator.
    type Mutator: Mutate<D>;

    /// Returns a reference to the generator.
    fn generator(&self) -> &Self::Generator;

    /// Returns a mutable reference to the generator.
    fn generator_mut(&mut self) -> &mut Self::Generator;

    /// Returns a reference to the selector.
    fn selector(&self) -> &Self::Selector;

    /// Returns a mutable reference to the selector.
    fn selector_mut(&mut self) -> &mut Self::Selector;

    /// Returns a reference to the crossover operator.
    fn cross_over(&self) -> &Self::CrossOver;

    /// Returns a mutable reference to the crossover operator.
    fn cross_over_mut(&mut self) -> &mut Self::CrossOver;

    /// Returns a reference to the mutator.
    fn mutator(&self) -> &Self::Mutator;

    /// Returns a mutable reference to the mutator.
    fn mutator_mut(&mut self) -> &mut Self::Mutator;
}

/// NSGA-II strategy.
#[derive(Debug)]
pub struct Nsga2Strategy<D, G, S, C, M> {
    generator: G,
    selector: S,
    cross_over: C,
    mutator: M,
    _param_domain: PhantomData<D>,
}

impl<D> Default for Nsga2Strategy<D, RandomGenerator, TournamentSelector, Exchange, Replace>
where
    D: Domain + Distribution<<D as Domain>::Point>,
{
    fn default() -> Self {
        Self::new(
            RandomGenerator,
            TournamentSelector::default(),
            Exchange::default(),
            Replace::default(),
        )
    }
}

impl<D, G, S, C, M> Nsga2Strategy<D, G, S, C, M>
where
    D: Domain,
    G: Generate<D>,
    S: Select<D>,
    C: CrossOver<D>,
    M: Mutate<D>,
{
    /// Makes a new `Nsga2Strategy` instance.
    pub fn new(generator: G, selector: S, cross_over: C, mutator: M) -> Self {
        Self {
            generator,
            selector,
            cross_over,
            mutator,
            _param_domain: PhantomData,
        }
    }
}

impl<D, G, S, C, M> Strategy<D> for Nsga2Strategy<D, G, S, C, M>
where
    D: Domain,
    G: Generate<D>,
    S: Select<D>,
    C: CrossOver<D>,
    M: Mutate<D>,
{
    type Generator = G;
    type Selector = S;
    type CrossOver = C;
    type Mutator = M;

    fn generator(&self) -> &Self::Generator {
        &self.generator
    }

    fn generator_mut(&mut self) -> &mut Self::Generator {
        &mut self.generator
    }

    fn selector(&self) -> &Self::Selector {
        &self.selector
    }

    fn selector_mut(&mut self) -> &mut Self::Selector {
        &mut self.selector
    }

    fn cross_over(&self) -> &Self::CrossOver {
        &self.cross_over
    }

    fn cross_over_mut(&mut self) -> &mut Self::CrossOver {
        &mut self.cross_over
    }

    fn mutator(&self) -> &Self::Mutator {
        &self.mutator
    }

    fn mutator_mut(&mut self) -> &mut Self::Mutator {
        &mut self.mutator
    }
}

/// [NSGA-II] based optimizer.
///
/// [NSGA-II]: https://ieeexplore.ieee.org/document/996017
#[derive(Debug)]
pub struct Nsga2Optimizer<P, S>
where
    P: Domain,
{
    population_size: usize,
    parent_population: Vec<Obs<P::Point, Vec<f64>>>,
    current_population: Vec<Obs<P::Point, Vec<f64>>>,
    strategy: S,
    param_domain: P,
    eval_queue: VecDeque<Obs<P::Point>>,
}

impl<P, S> Nsga2Optimizer<P, S>
where
    P: Domain,
    P::Point: Clone,
    S: Strategy<P>,
{
    /// Makes a new `Nsga2Optimizer` instance.
    pub fn new(param_domain: P, population_size: usize, strategy: S) -> Result<Self> {
        track_assert!(population_size >= 2, ErrorKind::InvalidInput; population_size);
        Ok(Self {
            population_size,
            parent_population: Vec::new(),
            current_population: Vec::new(),
            strategy,
            param_domain,
            eval_queue: VecDeque::new(),
        })
    }

    fn create_root_individual(&mut self, mut rng: impl Rng, mut idg: impl IdGen) -> Result<()> {
        let params = track!(self
            .strategy
            .generator_mut()
            .generate(&mut rng, &self.param_domain))?;
        self.eval_queue
            .push_back(track!(Obs::new(&mut idg, params))?);
        Ok(())
    }

    fn create_offspring_individual(
        &mut self,
        mut rng: impl Rng,
        mut idg: impl IdGen,
    ) -> Result<()> {
        let selector = self.strategy.selector_mut();
        let p0 = track!(selector.select(&mut rng, &self.parent_population))?;
        let p1 = track!(selector.select(&mut rng, &self.parent_population))?;

        let cross_over = self.strategy.cross_over_mut();
        let mut c0 = p0.param.clone();
        let mut c1 = p1.param.clone();
        track!(cross_over.cross_over(&mut rng, &mut c0, &mut c1))?;

        let mutator = self.strategy.mutator_mut();
        track!(mutator.mutate(&mut rng, &self.param_domain, &mut c0))?;
        track!(mutator.mutate(&mut rng, &self.param_domain, &mut c1))?;

        self.eval_queue.push_back(track!(Obs::new(&mut idg, c0))?);
        self.eval_queue.push_back(track!(Obs::new(&mut idg, c1))?);
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn fast_non_dominated_sort(
        &self,
        mut population: Vec<Obs<P::Point, Vec<f64>>>,
    ) -> Result<Vec<Vec<Obs<P::Point, Vec<f64>>>>> {
        let mut dominated_count = HashMap::new();
        let mut dominates_list = HashMap::new();

        for p in population.iter() {
            let mut sp = HashSet::new();
            let mut np = 0;

            for q in population.iter() {
                if track!(dominates(p, q))? {
                    sp.insert(q.id);
                } else if track!(dominates(q, p))? {
                    np += 1;
                }
            }
            dominated_count.insert(p.id, np);
            dominates_list.insert(p.id, sp);
        }

        let mut population_per_rank = Vec::new();
        while !population.is_empty() {
            let mut non_dominated_population = Vec::new();
            let mut i = 0;
            while i < population.len() {
                if dominated_count[&population[i].id] == 0 {
                    non_dominated_population.push(population.swap_remove(i));
                } else {
                    i += 1;
                }
            }

            for p in &non_dominated_population {
                for q in &dominates_list[&p.id] {
                    let nq = track_assert_some!(dominated_count.get_mut(q), ErrorKind::Bug);
                    *nq -= 1;
                }
            }

            track_assert!(!non_dominated_population.is_empty(), ErrorKind::Bug);
            population_per_rank.push(non_dominated_population);
        }

        Ok(population_per_rank)
    }

    fn crowding_distance_sort(&self, population: &mut [Obs<P::Point, Vec<f64>>]) {
        let l = population.len();
        let mut distances = HashMap::new();
        for i in 0..population[0].value.len() {
            population.sort_by_key(|x| OrderedFloat(x.value[i]));

            distances.insert(population[0].id, INFINITY);
            distances.insert(population[l - 1].id, INFINITY);
            let min = population[0].value[i];
            let max = population[l - 1].value[i];
            let width = max - min;

            for xs in population.windows(3) {
                let d = distances.entry(xs[1].id).or_insert(0.0);
                *d += (xs[2].value[i] - xs[0].value[i]) / width;
            }
        }

        population.sort_by_key(|x| OrderedFloat(distances[&x.id]));
        population.reverse();
    }
}

impl<P, S> Optimizer for Nsga2Optimizer<P, S>
where
    P: Domain,
    P::Point: Clone,
    S: Strategy<P>,
{
    type Param = P::Point;
    type Value = Vec<f64>;

    fn ask<R: Rng, G: IdGen>(&mut self, rng: R, idg: G) -> Result<Obs<Self::Param>> {
        if let Some(obs) = self.eval_queue.pop_front() {
            return Ok(obs);
        }

        if self.current_population.len() >= self.population_size {
            let population = self
                .parent_population
                .drain(..)
                .chain(self.current_population.drain(..))
                .collect::<Vec<_>>();
            let population_per_rank = track!(self.fast_non_dominated_sort(population))?;

            for mut population in population_per_rank {
                if self.parent_population.len() + population.len() < self.population_size {
                    self.parent_population.extend(population);
                } else {
                    let n = self.population_size - self.parent_population.len();
                    self.crowding_distance_sort(&mut population[..]);
                    self.parent_population
                        .extend(population.into_iter().take(n));
                    break;
                }
            }
        }

        if self.parent_population.is_empty() {
            track!(self.create_root_individual(rng, idg))?;
        } else {
            track!(self.create_offspring_individual(rng, idg))?;
        }
        Ok(track_assert_some!(
            self.eval_queue.pop_front(),
            ErrorKind::Bug
        ))
    }

    fn tell(&mut self, obs: Obs<Self::Param, Self::Value>) -> Result<()> {
        self.current_population.push(obs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::DiscreteDomain;
    use crate::generators::SerialIdGenerator;
    use rand;
    use trackable::result::TestResult;

    #[test]
    fn nsga2_works() -> TestResult {
        let param_domain = track!(DiscreteDomain::new(10))?;
        let population_size = 10;
        let strategy = Nsga2Strategy::default();
        let mut opt = track!(Nsga2Optimizer::new(param_domain, population_size, strategy))?;
        let mut rng = rand::thread_rng();
        let mut idg = SerialIdGenerator::new();

        let obs = track!(opt.ask(&mut rng, &mut idg))?;
        track!(opt.tell(obs.map_value(|()| vec![1.0])))?;

        Ok(())
    }
}
