//! Budget for evaluating parameters.

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

/// An object with a specific budget.
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
