use pyo3::prelude::*;

use ordered_float::OrderedFloat;
use pathfinding::directed::edmonds_karp::{
    SparseCapacity,
    Edge,
    edmonds_karp
};

use std::time::Instant;
use log::info;

use crate::lowtime_graph::LowtimeGraph;
use crate::utils;


#[pyclass]
pub struct PhillipsDessouky {
   graph: LowtimeGraph,
}

#[pymethods]
impl PhillipsDessouky {
    #[new]
    fn new(
        node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges: Vec<((u32, u32), f64)>,
    ) -> PyResult<Self> {
        Ok(PhillipsDessouky {
            graph: LowtimeGraph::of_python(
                node_ids,
                source_node_id,
                sink_node_id,
                edges,
            )
        })
    }

    fn max_flow(&self) -> PyResult<Vec<((u32, u32), f64)>> {
        // TODO(ohjun): temporary for compile check
        Ok(Vec::new())
        // let profiling_start = Instant::now();
        // let edges_edmonds_karp: Vec<Edge<u32, OrderedFloat<f64>>> = self.edges_raw.iter().map(|((from, to), cap)| {
        //     ((*from, *to), OrderedFloat(*cap))
        // }).collect();
        // let profiling_end = Instant::now();
        // info!("PROFILING Rust_PhillipsDessouky::max_flow scaling to OrderedFloat time: {:.10}s", utils::profile_duration(profiling_start, profiling_end));

        // let profiling_start = Instant::now();
        // let (flows, _max_flow, _min_cut) = edmonds_karp::<_, _, _, SparseCapacity<_>>(
        //     &self.node_ids,
        //     &self.source_node_id,
        //     &self.sink_node_id,
        //     edges_edmonds_karp,
        // );
        // let profiling_end = Instant::now();
        // info!("PROFILING Rust_PhillipsDessouky::max_flow edmonds_karp time: {:.10}s", utils::profile_duration(profiling_start, profiling_end));

        // let profiling_start = Instant::now();
        // let flows_f64: Vec<((u32, u32), f64)> = flows.iter().map(|((from, to), flow)| {
        //     ((*from, *to), flow.into_inner())
        // }).collect();
        // let profiling_end = Instant::now();
        // info!("PROFILING Rust_PhillipsDessouky::max_flow reformat OrderedFloat to f64 time: {:.10}s", utils::profile_duration(profiling_start, profiling_end));

        // Ok(flows_f64)
    }
}
