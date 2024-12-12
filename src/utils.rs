use std::time::{
    Instant,
    Duration,
};

// This function is not used in the codebase, but it is left here
// to facilitate profiling during development.
pub fn get_duration(start: Instant, end: Instant) -> f64 {
    let duration: Duration = end.duration_since(start);
    let seconds = duration.as_secs();
    let subsec_nanos = duration.subsec_nanos();

    let fractional_seconds = subsec_nanos as f64 / 1_000_000_000.0;
    let total_seconds = seconds as f64 + fractional_seconds;

    return total_seconds;
}
