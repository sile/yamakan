use crate::optimizer::Optimizer;
use crate::space::ParamSpace;
use rand;
use rand::distributions::Distribution;
use rand::Rng;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct RandomOptimizer<P, V> {
    param_space: P,
    _value: PhantomData<V>,
}
impl<P, V> RandomOptimizer<P, V>
where
    P: ParamSpace<Internal = f64>,
{
    pub fn new(param_space: P) -> Self {
        Self {
            param_space,
            _value: PhantomData,
        }
    }

    pub fn param_space(&self) -> &P {
        &self.param_space
    }
}
impl<P, V> Optimizer for RandomOptimizer<P, V>
where
    P: ParamSpace<Internal = f64>,
{
    type Param = P::External;
    type Value = V;

    fn ask<R: Rng>(&mut self, rng: &mut R) -> Self::Param {
        let r = self.param_space.internal_range();
        let mu = 0.5 * (r.start + r.end);
        let sigma = r.end - r.start;
        let d = rand::distributions::Normal::new(mu, sigma);
        loop {
            let draw = d.sample(rng);
            if r.start <= draw && draw < r.end {
                return self.param_space.externalize(&draw);
            }
        }
    }

    fn tell(&mut self, _param: Self::Param, _value: Self::Value) {}
}
impl<P, V> Default for RandomOptimizer<P, V>
where
    P: Default + ParamSpace<Internal = f64>,
{
    fn default() -> Self {
        Self::new(P::default())
    }
}
