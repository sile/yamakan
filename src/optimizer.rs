pub trait Optimizer {
    type Param;
    type Value;
    fn ask(&mut self) -> Option<Self::Param>;
    fn tell(&mut self, param: Self::Param, value: Self::Value);
}
