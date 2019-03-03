pub trait Optimizer {
    type Param;
    type Value;
    fn ask(&mut self) -> Self::Param;
    fn tell(&mut self, param: Self::Param, value: Self::Value);
}
