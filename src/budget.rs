//! Budget for evaluating parameters.
use crate::{ErrorKind, Result};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std;
use std::cmp::{Ordering, Reverse};

/// Budget.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Budget {
    consumption: u64,
    soft_limit: u64,
    hard_limit: u64,
}
impl Budget {
    /// Makes a new `Budget` instance which has the given amount (limit) of budget.
    pub const fn new(limit: u64) -> Self {
        Self {
            consumption: 0,
            soft_limit: limit,
            hard_limit: limit,
        }
    }

    /// Returns the hard limit of this budget.
    pub const fn hard_limit(&self) -> u64 {
        self.hard_limit
    }

    /// Returns the soft limit of this budget.
    pub const fn soft_limit(&self) -> u64 {
        self.soft_limit
    }

    /// Sets the hard limit of this budget.
    ///
    /// # Errors
    ///
    /// `limit` must be greater than or equal to the soft limit and consumption,
    /// otherwise an `ErrorKind::InvalidInput` error will be returned.
    pub fn set_hard_limit(&mut self, limit: u64) -> Result<()> {
        track_assert!(self.soft_limit <= limit, ErrorKind::InvalidInput; limit, self.soft_limit);
        track_assert!(self.consumption <= limit, ErrorKind::InvalidInput; limit, self.consumption);
        self.hard_limit = limit;
        Ok(())
    }

    /// Sets the soft limit of this budget.
    ///
    /// # Errors
    ///
    /// If `limit` exceeded the hard limit of this budget, an `ErrorKind::InvalidInput` error will be returned.
    pub fn set_soft_limit(&mut self, limit: u64) -> Result<()> {
        track_assert!(limit <= self.hard_limit, ErrorKind::InvalidInput; limit, self.hard_limit);
        self.soft_limit = limit;
        Ok(())
    }

    /// Consumes the given amount of this budget.
    ///
    /// # Errors
    ///
    /// If the hard limit of the budget is exceeded, an `ErrorKind::InvalidInput` error will be returned.
    ///
    /// Note that it is allowed to consume over the soft limit of the budget.
    pub fn consume(&mut self, amount: u64) -> Result<()> {
        track_assert!(self.consumption + amount <= self.hard_limit, ErrorKind::InvalidInput;
                      self.consumption, amount, self.hard_limit);
        self.consumption += amount;
        Ok(())
    }

    /// Sets the consumption of this budget.
    ///
    /// # Errors
    ///
    /// `amount` must be a value between `[self.consumption()..self.hard_limit()]`,
    /// otherwise an `ErrorKind::InvalidInput` error will be returned.
    pub fn set_consumption(&mut self, amount: u64) -> Result<()> {
        track_assert!(self.consumption <= amount, ErrorKind::InvalidInput; self.consumption, amount);
        track_assert!(amount <= self.hard_limit, ErrorKind::InvalidInput; amount, self.hard_limit);

        self.consumption = amount;
        Ok(())
    }

    /// Returns the total consumption of this budget.
    pub const fn consumption(&self) -> u64 {
        self.consumption
    }

    /// Returns the remaining amount for the hard limit of this budget.
    pub const fn hard_remaining(&self) -> u64 {
        self.hard_limit - self.consumption
    }

    /// Returns the remaining amount for the soft limit of this budget.
    ///
    /// # Errors
    ///
    /// If the consumption of the budget exceeded the soft limit, `Err(excess amount)` will be returned.
    pub fn soft_remaining(&self) -> std::result::Result<u64, u64> {
        if self.consumption <= self.soft_limit {
            Ok(self.soft_limit - self.consumption)
        } else {
            Err(self.consumption - self.soft_limit)
        }
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
