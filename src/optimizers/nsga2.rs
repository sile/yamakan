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
        parent_count: usize,
    ) -> Result<Vec<&'a Obs<D::Point, Vec<f64>>>> {
        (0..parent_count)
            .map(|_| track!(self.select(&mut rng, population)))
            .collect()
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

pub trait Variator<D: Domain> {
    fn evolve<R: Rng>(
        &mut self,
        rng: R,
        parents: &[&Obs<D::Point, Vec<f64>>],
    ) -> Result<Vec<D::Point>>;
}

fn dominates<P>(a: &Obs<P, Vec<f64>>, b: &Obs<P, Vec<f64>>) -> Result<bool> {
    track_assert_eq!(a.value.len(), b.value.len(), ErrorKind::InvalidInput);
    if a.value.iter().zip(b.value.iter()).any(|(a, b)| a > b) {
        Ok(false)
    } else {
        Ok(a.value.iter().zip(b.value.iter()).any(|(a, b)| a < b))
    }
}

// TODO: remove
// SSX: Subsequence exchange crossover.

pub trait Strategy<D: Domain> {
    type Generator: Generate<D>;
    type Selector: Select<D>;
    type Variator: Variator<D>;

    fn generator(&self) -> &Self::Generator;
    fn generator_mut(&mut self) -> &mut Self::Generator;

    fn selector(&self) -> &Self::Selector;
    fn selector_mut(&mut self) -> &mut Self::Selector;

    fn variator(&self) -> &Self::Variator;
    fn variator_mut(&mut self) -> &mut Self::Variator;
}

#[derive(Debug)]
pub struct Nsga2Strategy<D, G, S, V> {
    generator: G,
    selector: S,
    variator: V,
    _param_domain: PhantomData<D>,
}

impl<D, G, S, V> Nsga2Strategy<D, G, S, V>
where
    D: Domain,
    G: Generate<D>,
    S: Select<D>,
    V: Variator<D>,
{
    pub fn new(generator: G, selector: S, variator: V) -> Self {
        Self {
            generator,
            selector,
            variator,
            _param_domain: PhantomData,
        }
    }
}

impl<D, G, S, V> Strategy<D> for Nsga2Strategy<D, G, S, V>
where
    D: Domain,
    G: Generate<D>,
    S: Select<D>,
    V: Variator<D>,
{
    type Generator = G;
    type Selector = S;
    type Variator = V;

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

    fn variator(&self) -> &Self::Variator {
        &self.variator
    }

    fn variator_mut(&mut self) -> &mut Self::Variator {
        &mut self.variator
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
        let parents = track!(self.strategy.selector_mut().select_parents(
            &mut rng,
            &self.parent_population,
            2
        ))?;
        for params in track!(self.strategy.variator_mut().evolve(&mut rng, &parents))? {
            self.eval_queue
                .push_back(track!(Obs::new(&mut idg, params))?);
        }
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
