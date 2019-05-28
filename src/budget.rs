//! Budget for evaluating parameters.
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std;

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
