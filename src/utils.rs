use std::time::{
    Instant,
    Duration,
};

pub fn profile_duration(start: Instant, end: Instant) -> f64 {
    let duration: Duration = end.duration_since(start);
    let seconds = duration.as_secs();
    let subsec_nanos = duration.subsec_nanos();

    let fractional_seconds = subsec_nanos as f64 / 1_000_000_000.0;
    let total_seconds = seconds as f64 + fractional_seconds;

    return total_seconds;
}