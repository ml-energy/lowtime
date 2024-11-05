use pyo3::prelude::*;

use ordered_float::OrderedFloat;
use pathfinding::directed::edmonds_karp::{
    SparseCapacity,
    Edge,
    edmonds_karp
};

use std::time::{
    Instant,
    Duration,
};
use log::info;


#[pyclass]
struct PhillipsDessouky {
    #[pyo3(get)]
    node_ids: Vec<u32>,
    source_node_id: u32,
    sink_node_id: u32,
    edges_raw: Vec<((u32, u32), f64)>,
}

#[pymethods]
impl PhillipsDessouky {
    #[new]
    fn new(
        node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges_raw: Vec<((u32, u32), f64)>,
    ) -> PyResult<Self> {
        Ok(PhillipsDessouky {
            node_ids,
            source_node_id,
            sink_node_id,
            edges_raw,
        })
    }

    fn max_flow(&self) -> PyResult<Vec<((u32, u32), f64)>> {
        let profiling_start = Instant::now();
        let edges_edmonds_karp: Vec<Edge<u32, OrderedFloat<f64>>> = self.edges_raw.iter().map(|((from, to), cap)| {
            ((*from, *to), OrderedFloat(*cap))
        }).collect();
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessouky::max_flow scaling to OrderedFloat time: {:.10}s", PhillipsDessouky::profile_duration(profiling_start, profiling_end));

        let profiling_start = Instant::now();
        let (flows, _max_flow, _min_cut) = edmonds_karp::<_, _, _, SparseCapacity<_>>(
            &self.node_ids,
            &self.source_node_id,
            &self.sink_node_id,
            edges_edmonds_karp,
        );
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessouky::max_flow edmonds_karp time: {:.10}s", PhillipsDessouky::profile_duration(profiling_start, profiling_end));

        let profiling_start = Instant::now();
        let flows_f64: Vec<((u32, u32), f64)> = flows.iter().map(|((from, to), flow)| {
            ((*from, *to), flow.into_inner())
        }).collect();
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessouky::max_flow reformat OrderedFloat to f64 time: {:.10}s", PhillipsDessouky::profile_duration(profiling_start, profiling_end));

        Ok(flows_f64)
    }
}

// Private to Rust, not exposed to Python
impl PhillipsDessouky {
    fn profile_duration(start: Instant, end: Instant) -> f64 {
        let duration: Duration = end.duration_since(start);
        let seconds = duration.as_secs();
        let subsec_nanos = duration.subsec_nanos();

        let fractional_seconds = subsec_nanos as f64 / 1_000_000_000.0;
        let total_seconds = seconds as f64 + fractional_seconds;

        return total_seconds;
    }
}

// A Python module implemented in Rust.
#[pymodule]
fn lowtime_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();  // send Rust logs to Python logger
    m.add_class::<PhillipsDessouky>()?;
    Ok(())
}
