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
