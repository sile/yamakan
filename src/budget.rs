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

    // fn remaining()
}

#[derive(Debug, Clone)]
pub struct Budgeted<T> {
    budget: Budget,
    value: T,
}
impl<T> Budgeted<T> {
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    pub fn get(&self) -> &T {
        &self.value
    }
}
