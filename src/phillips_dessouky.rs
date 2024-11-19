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
        edges_raw: Vec<((u32, u32), (f64, f64, f64, f64), Option<(bool, u64, u64, u64, u64, u64, u64, u64)>)>,
    ) -> PyResult<Self> {
        Ok(PhillipsDessouky {
            graph: LowtimeGraph::of_python(
                node_ids,
                source_node_id,
                sink_node_id,
                edges_raw,
            )
        })
    }

    fn max_flow(&self) -> Vec<((u32, u32), f64)> {
        self.graph.max_flow()
    }
}
