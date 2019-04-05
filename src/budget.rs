#[derive(Debug, Clone)]
pub struct Budget {
    consumption: u64,
    amount: u64,
}
impl Budget {
    pub fn new(amount: u64) -> Self {
        Self {
            consumption: 0,
            amount,
        }
    }

    pub fn consume(&mut self, n: u64) {
        self.consumption += n;
    }

    pub fn consumption(&self) -> u64 {
        self.consumption
    }

    pub fn amount(&self) -> u64 {
        self.amount
    }

    pub fn set_amount(&mut self, n: u64) {
        self.amount = n;
    }

    pub fn remaining(&self) -> u64 {
        if self.amount < self.consumption {
            0
        } else {
            self.amount - self.consumption
        }
    }

    pub fn excess(&self) -> u64 {
        if self.consumption > self.amount {
            self.consumption - self.amount
        } else {
            0
        }
    }
}

#[derive(Debug, Clone)]
pub struct Budgeted<T> {
    budget: Budget,
    inner: T,
}
impl<T> Budgeted<T> {
    pub fn new(budget: Budget, inner: T) -> Self {
        Budgeted { budget, inner }
    }

    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    pub fn budget_mut(&mut self) -> &mut Budget {
        &mut self.budget
    }

    pub fn get(&self) -> &T {
        &self.inner
    }
}
