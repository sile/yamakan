//! Budget for evaluating parameters.
use std::cmp::{Ordering, Reverse};

/// Budget.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    consumption: u64,
    amount: u64,
}
impl Budget {
    /// Makes a new `Budget` instance which has the given amount of budget.
    pub const fn new(amount: u64) -> Self {
        Self {
            consumption: 0,
            amount,
        }
    }

    /// Returns the total amount of this budget.
    pub const fn amount(&self) -> u64 {
        self.amount
    }

    /// Sets the total amount of this budget.
    pub fn set_amount(&mut self, amount: u64) {
        self.amount = amount;
    }

    /// Consumes the given amount of this budget.
    ///
    /// Note that it is allowed to consume over the total amount of the budget.
    pub fn consume(&mut self, amount: u64) {
        self.consumption += amount;
    }

    /// Returns the total consumption of this budget.
    pub const fn consumption(&self) -> u64 {
        self.consumption
    }

    /// Returns the remaining amount of this budget.
    pub const fn remaining(&self) -> i64 {
        self.amount as i64 - self.consumption as i64
    }
}

/// An object which has a specific budget.
#[derive(Debug, Clone, Copy)]
pub struct Budgeted<T> {
    budget: Budget,
    inner: T,
}
impl<T> Budgeted<T> {
    /// Makes a new `Budgeted` instance.
    pub const fn new(budget: Budget, inner: T) -> Self {
        Budgeted { budget, inner }
    }

    /// Returns the budget of this instance.
    pub const fn budget(&self) -> Budget {
        self.budget
    }

    /// Returns a mutable reference to the budget of this instance.
    pub fn budget_mut(&mut self) -> &mut Budget {
        &mut self.budget
    }

    /// Returns a reference to the underlying object.
    pub const fn get(&self) -> &T {
        &self.inner
    }

    /// Returns a mutable reference to the underlying object.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consumes the `Budgeted`, returning the wrapped object.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

/// An object leveled by its resource consumption.
///
/// Note that this is provided with the intention of being used for optimizations of minimizing direction.
/// That is, the higher the level, the lower the order by `std::cmp::Ordering`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Leveled<T> {
    level: u64,
    inner: T,
}
impl<T> Leveled<T> {
    /// Makes a new `Leveled` instance.
    pub const fn new(level: u64, inner: T) -> Self {
        Self { level, inner }
    }

    /// Returns the level of this instance.
    pub const fn level(&self) -> u64 {
        self.level
    }

    /// Returns a reference to the underlying object.
    pub const fn get(&self) -> &T {
        &self.inner
    }

    /// Returns a mutable reference to the underlying object.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consumes the `Leveled`, returning the wrapped object.
    pub fn into_inner(self) -> T {
        self.inner
    }

    fn to_tuple(&self) -> (Reverse<u64>, &T) {
        (Reverse(self.level), &self.inner)
    }
}
impl<T: PartialOrd> PartialOrd for Leveled<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_tuple().partial_cmp(&other.to_tuple())
    }
}
impl<T: Ord> Ord for Leveled<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_tuple().cmp(&other.to_tuple())
    }
}
