//! NSGA-II (Non-dominated Sorting Genetic Algorithm II).
//!
//! # References
//!
//! - [A fast and elitist multiobjective genetic algorithm: NSGA-II][NSGA-II]
//!
//! [NSGA-II]: https://ieeexplore.ieee.org/document/996017
#![allow(missing_docs)] // TODO
use crate::{Domain, ErrorKind, IdGen, Obs, Optimizer, Result};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::INFINITY;
use std::marker::PhantomData;

pub trait Generate<D: Domain> {
    fn generate<R: Rng>(&mut self, rng: R, domain: &D) -> Result<D::Point>;
}

#[derive(Debug)]
pub struct RandomGenerator;

impl<D> Generate<D> for RandomGenerator
where
    D: Domain + Distribution<<D as Domain>::Point>,
{
    fn generate<R: Rng>(&mut self, mut rng: R, domain: &D) -> Result<D::Point> {
        Ok(domain.sample(&mut rng))
    }
}

pub trait Select<D: Domain> {
    fn select_parents<'a, R: Rng>(
        &mut self,
        mut rng: R,
        population: &'a [Obs<D::Point, Vec<f64>>],
    ) -> Result<(&'a Obs<D::Point, Vec<f64>>, &'a Obs<D::Point, Vec<f64>>)> {
        let p0 = track!(self.select(&mut rng, population))?;
        let p1 = track!(self.select(&mut rng, population))?;
        Ok((p0, p1))
    }

    fn select<'a, R: Rng>(
        &mut self,
        rng: R,
        population: &'a [Obs<D::Point, Vec<f64>>],
    ) -> Result<&'a Obs<D::Point, Vec<f64>>>;
}

#[derive(Debug)]
pub struct TournamentSelector {
    tournament_size: usize,
}

impl TournamentSelector {
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

impl Default for TournamentSelector {
    fn default() -> Self {
        Self::new(2)
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

pub trait CrossOver<D: Domain> {
    fn cross_over<R: Rng>(
        &mut self,
        rng: R,
        p0: D::Point,
        p1: D::Point,
    ) -> Result<(D::Point, D::Point)>;
}

pub trait Mutate<D: Domain> {
    fn mutate<R: Rng>(&mut self, rng: R, domain: &D, p: D::Point) -> Result<D::Point>;
}

#[derive(Debug)]
pub struct Exchange {
    probability: f64,
}

impl Exchange {
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
        p0: D::Point,
        p1: D::Point,
    ) -> Result<(D::Point, D::Point)> {
        if rng.gen_bool(self.probability) {
            Ok((p1, p0))
        } else {
            Ok((p0, p1))
        }
    }
}

#[derive(Debug)]
pub struct Replace {
    probability: f64,
}

impl Replace {
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
    fn mutate<R: Rng>(&mut self, mut rng: R, domain: &D, p: D::Point) -> Result<D::Point> {
        if rng.gen_bool(self.probability) {
            Ok(domain.sample(&mut rng))
        } else {
            Ok(p)
        }
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

pub trait Strategy<D: Domain> {
    type Generator: Generate<D>;
    type Selector: Select<D>;
    type CrossOver: CrossOver<D>;
    type Mutator: Mutate<D>;

    fn generator(&self) -> &Self::Generator;
    fn generator_mut(&mut self) -> &mut Self::Generator;

    fn selector(&self) -> &Self::Selector;
    fn selector_mut(&mut self) -> &mut Self::Selector;

    fn cross_over(&self) -> &Self::CrossOver;
    fn cross_over_mut(&mut self) -> &mut Self::CrossOver;

    fn mutator(&self) -> &Self::Mutator;
    fn mutator_mut(&mut self) -> &mut Self::Mutator;
}

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
        let (p0, p1) = track!(self
            .strategy
            .selector_mut()
            .select_parents(&mut rng, &self.parent_population))?;
        let (c0, c1) = track!(self.strategy.cross_over_mut().cross_over(
            &mut rng,
            p0.param.clone(),
            p1.param.clone()
        ))?;
        let c0 = track!(self
            .strategy
            .mutator_mut()
            .mutate(&mut rng, &self.param_domain, c0))?;
        let c1 = track!(self
            .strategy
            .mutator_mut()
            .mutate(&mut rng, &self.param_domain, c1))?;
        self.eval_queue.push_back(track!(Obs::new(&mut idg, c0))?);
        self.eval_queue.push_back(track!(Obs::new(&mut idg, c1))?);
        Ok(())
    }

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
