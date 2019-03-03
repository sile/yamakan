pub fn linspace(low: f64, high: f64, num: usize) -> impl Iterator<Item = f64> {
    let delta = (high - low) / num as f64;
    (0..num).map(move |i| low + delta * (i as f64))
}
