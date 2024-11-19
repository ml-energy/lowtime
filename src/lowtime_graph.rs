use std::collections::HashMap;

use ordered_float::OrderedFloat;
use pathfinding::directed::edmonds_karp::{
    SparseCapacity,
    Edge,
    edmonds_karp
};

use std::time::Instant;
use log::info;

use crate::operation::Operation;
use crate::utils;


pub struct LowtimeGraph {
    node_ids: Vec<u32>,
    source_node_id: u32,
    sink_node_id: u32,
    edges: HashMap<u32, HashMap<u32, LowtimeEdge>>,
    preds: HashMap<u32, u32>,
}

impl LowtimeGraph {
    pub fn new() -> Self {
        LowtimeGraph {
            node_ids: Vec::new(),
            source_node_id: u32::MAX,
            sink_node_id: u32::MAX,
            edges: HashMap::new(),
            preds: HashMap::new(),
        }
    }

    pub fn of_python(
        node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges_raw: Vec<((u32, u32), (f64, f64, f64, f64), Option<(bool, u64, u64, u64, u64, u64, u64, u64)>)>,
    ) -> Self {
        let mut graph = LowtimeGraph::new();
        // TODO(ohjun): this is very bad. will make better after checking that minimal change version works.
        graph.node_ids = node_ids;
        graph.source_node_id = source_node_id;
        graph.sink_node_id = sink_node_id;
        edges_raw.iter().for_each(|(
                (from, to),
                (capacity, flow, ub, lb),
                op_details,
            )| {
            let op = op_details.map(|(
              is_dummy,
              duration,
              max_duration,
              min_duration,
              earliest_start,
              latest_start,
              earliest_finish,
              latest_finish,
            )| Operation::new(
                is_dummy,
                duration,
                max_duration,
                min_duration,
                earliest_start,
                latest_start,
                earliest_finish,
                latest_finish
            ));
            graph.add_edge(*from, *to, LowtimeEdge::new(op, *capacity, *flow, *ub, *lb))
        });
        graph
    }

    pub fn get_node_ids(&self) -> Vec<u32> {
        let mut node_ids: Vec<u32> = self.edges.keys().cloned().collect();
        node_ids.sort();
        node_ids
    }

    pub fn add_edge(&mut self, from: u32, to: u32, edge: LowtimeEdge) -> () {
        self.edges
            .entry(from)
            .or_insert_with(HashMap::new)
            .insert(to, edge);
        self.preds.insert(to, from);
    }

    pub fn get_edge(&self, from: u32, to: u32) -> Option<&LowtimeEdge> {
        self.edges
            .get(&from)
            .and_then(|to_edges| to_edges.get(&to))
    }

    pub fn get_mut_op(&mut self, from: u32, to: u32) -> Option<&mut LowtimeEdge> {
        self.edges
            .get_mut(&from)
            .and_then(|to_edges| to_edges.get_mut(&to))
    }

    pub fn max_flow(&self) -> Vec<((u32, u32), f64)> {
        let profiling_start = Instant::now();
        let edges_edmonds_karp: Vec<Edge<u32, OrderedFloat<f64>>> = self.edges.iter().flat_map(|(from, inner)|
            inner.iter().map(|(to, edge)| ((*from, *to), edge.get_capacity()))
        ).collect();
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessouky::max_flow scaling to OrderedFloat time: {:.10}s", utils::profile_duration(profiling_start, profiling_end));

        let profiling_start = Instant::now();
        let (flows, _max_flow, _min_cut) = edmonds_karp::<_, _, _, SparseCapacity<_>>(
            &self.node_ids,
            &self.source_node_id,
            &self.sink_node_id,
            edges_edmonds_karp,
        );
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessouky::max_flow edmonds_karp time: {:.10}s", utils::profile_duration(profiling_start, profiling_end));

        let profiling_start = Instant::now();
        let flows_f64: Vec<((u32, u32), f64)> = flows.iter().map(|((from, to), flow)| {
            ((*from, *to), flow.into_inner())
        }).collect();
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessouky::max_flow reformat OrderedFloat to f64 time: {:.10}s", utils::profile_duration(profiling_start, profiling_end));

        flows_f64
    }
}

pub struct LowtimeEdge {
    op: Option<Operation>,
    capacity: OrderedFloat<f64>,
    flow: OrderedFloat<f64>,
    ub: OrderedFloat<f64>,
    lb: OrderedFloat<f64>,
}

impl LowtimeEdge {
    fn new(
        op: Option<Operation>,
        capacity: f64,
        flow: f64,
        ub: f64,
        lb: f64,
    ) -> Self {
        LowtimeEdge {
            op,
            capacity: OrderedFloat(capacity),
            flow: OrderedFloat(flow),
            ub: OrderedFloat(ub),
            lb: OrderedFloat(lb),
        }
    }

    fn get_op(&self) -> Option<&Operation> {
        self.op.as_ref()
    }

    fn get_capacity(&self) -> OrderedFloat<f64> {
        self.capacity
    }

    fn get_flow(&self) -> OrderedFloat<f64> {
        self.flow
    }

    fn get_ub(&self) -> OrderedFloat<f64> {
        self.ub
    }

    fn get_lb(&self) -> OrderedFloat<f64> {
        self.lb
    }

}