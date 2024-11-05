use ordered_float::OrderedFloat;
use pathfinding::directed::edmonds_karp::{
    SparseCapacity,
    Edge,
    edmonds_karp
};

// for profiling
use std::time::{
    Instant,
    Duration,
};
use log::info;


// Private to Rust, not exposed to Python
pub struct PhillipsDessoukyImpl {
    node_ids: Vec<u32>,
    source_node_id: u32,
    sink_node_id: u32,
    edges_raw: Vec<((u32, u32), f64)>,
}

// Private to Rust, not exposed to Python
impl PhillipsDessoukyImpl {
    pub fn new(
        node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges_raw: Vec<((u32, u32), f64)>,
    ) -> Self {
        // Note: Intentionally nothing done here to profile data transfer latency
        PhillipsDessoukyImpl {
            node_ids,
            source_node_id,
            sink_node_id,
            edges_raw,
        }
    }

    pub fn max_flow(&self) -> Vec<((u32, u32), f64)> {
        let profiling_start = Instant::now();
        let edges_edmonds_karp: Vec<Edge<u32, OrderedFloat<f64>>> = self.edges_raw.iter().map(|((from, to), cap)| {
            ((*from, *to), OrderedFloat(*cap))
        }).collect();
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessoukyImpl::max_flow scaling to OrderedFloat time: {:.10}s", PhillipsDessoukyImpl::profile_duration(profiling_start, profiling_end));

        let profiling_start = Instant::now();
        let (flows, _max_flow, _min_cut) = edmonds_karp::<_, _, _, SparseCapacity<_>>(
            &self.node_ids,
            &self.source_node_id,
            &self.sink_node_id,
            edges_edmonds_karp,
        );
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessoukyImpl::max_flow edmonds_karp time: {:.10}s", PhillipsDessoukyImpl::profile_duration(profiling_start, profiling_end));

        let profiling_start = Instant::now();
        let flows_f64: Vec<((u32, u32), f64)> = flows.iter().map(|((from, to), flow)| {
            ((*from, *to), flow.into_inner())
        }).collect();
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessoukyImpl::max_flow reformat OrderedFloat to f64 time: {:.10}s", PhillipsDessoukyImpl::profile_duration(profiling_start, profiling_end));

        flows_f64
    }

    fn profile_duration(start: Instant, end: Instant) -> f64 {
        let duration: Duration = end.duration_since(start);
        let seconds = duration.as_secs();
        let subsec_nanos = duration.subsec_nanos();

        let fractional_seconds = subsec_nanos as f64 / 1_000_000_000.0;
        let total_seconds = seconds as f64 + fractional_seconds;

        return total_seconds;
    }
}