//! Budget for evaluating parameters.
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std;
use std::cmp::{Ordering, Reverse};

/// Budget.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Budget {
    /// The amount of this budget.
    pub amount: u64,

    /// The consumption of this budget.
    ///
    /// Note that this value can exceed the budget amount.
    pub consumption: u64,
}
impl Budget {
    /// Makes a new `Budget` instance which has the given amount of budget.
    pub const fn new(amount: u64) -> Self {
        Self {
            consumption: 0,
            amount,
        }
    }

    /// Returns the remaining amount of this budget.
    ///
    /// # Errors
    ///
    /// If the consumption of the budget exceeded the budget amount, `Err(excess amount)` will be returned.
    pub fn remaining(&self) -> std::result::Result<u64, u64> {
        if self.consumption <= self.amount {
            Ok(self.amount - self.consumption)
        } else {
            Err(self.consumption - self.amount)
        }
    }

    /// Returns `true` if the consumption has exceeded the budget amount, otherwise `false`.
    pub fn is_consumed(&self) -> bool {
        self.consumption >= self.amount
    }
}

/// An object which has a specific budget.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
