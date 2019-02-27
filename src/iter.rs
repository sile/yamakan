pub fn linspace(low: f64, high: f64, num: usize) -> Vec<f64> {
    let mut vs = Vec::with_capacity(num);
    let delta = (high - low) / num as f64;
    let mut v = low;
    for _ in 0..num {
        vs.push(v);
        v += delta;
        if v > high {
            v = high;
        }
    }
    vs
}
