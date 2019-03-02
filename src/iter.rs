pub fn linspace(low: f64, high: f64, num: usize) -> impl Iterator<Item = f64> {
    let delta = (high - low) / num as f64;
    (0..num).map(move |i| low + delta * (i as f64))
}

pub fn bincount<I>(xs: I, min_size: usize) -> Vec<f64>
where
    I: Iterator<Item = (usize, f64)>,
{
    let mut bins = vec![0.0; min_size];
    for (x, weight) in xs {
        bins.resize(x, 0.0);
        bins[x] += weight;
    }
    bins
}
