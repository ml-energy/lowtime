use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use ordered_float::OrderedFloat;
use pathfinding::directed::edmonds_karp::{
    SparseCapacity,
    Edge,
    EKFlows,
    edmonds_karp
};

use std::time::Instant;
use log::{info, debug, Level};

use crate::operation::Operation;
use crate::utils;


#[derive(Clone)]
pub struct LowtimeGraph {
    node_ids: Vec<u32>,
    source_node_id: Option<u32>,
    sink_node_id: Option<u32>,
    edges: HashMap<u32, HashMap<u32, LowtimeEdge>>,
    preds: HashMap<u32, HashSet<u32>>,
    num_edges: usize,
}

impl LowtimeGraph {
    pub fn new() -> Self {
        LowtimeGraph {
            node_ids: Vec::new(),
            source_node_id: None,
            sink_node_id: None,
            edges: HashMap::new(),
            preds: HashMap::new(),
            num_edges: 0,
        }
    }

    pub fn of_python(
        mut node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges_raw: Vec<((u32, u32), (f64, f64, f64, f64), Option<(bool, u64, u64, u64, u64, u64, u64, u64)>)>,
    ) -> Self {
        let mut graph = LowtimeGraph::new();
        node_ids.sort();
        graph.node_ids = node_ids.clone();
        graph.source_node_id = Some(source_node_id);
        graph.sink_node_id = Some(sink_node_id);

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

    pub fn max_flow(&self) -> EKFlows<u32, OrderedFloat<f64>> {
        let edges_edmonds_karp = self.get_ek_preprocessed_edges();

        // TESTING(ohjun)
        // info!("self.node_ids.len(): {}", self.node_ids.len());
        // info!("edges_edmonds_karp.len(): {}", edges_edmonds_karp.len());
        // info!("self.source_node_id.unwrap(): {}", self.source_node_id.unwrap());
        // info!("self.sink_node_id.unwrap(): {}", self.sink_node_id.unwrap());

        let profiling_start = Instant::now();
        let (flows, max_flow, min_cut) = edmonds_karp::<_, _, _, SparseCapacity<_>>(
            &self.node_ids,
            &self.source_node_id.unwrap(),
            &self.sink_node_id.unwrap(),
            edges_edmonds_karp,
        );
        let profiling_end = Instant::now();
        info!("PROFILING Rust_PhillipsDessouky::max_flow edmonds_karp time: {:.10}s", utils::profile_duration(profiling_start, profiling_end));

        (flows, max_flow, min_cut)
    }

    pub fn get_source_node_id(&self) -> u32 {
        self.source_node_id.unwrap()
    }

    pub fn set_source_node_id(&mut self, new_source_node_id: u32) -> () {
        self.source_node_id = Some(new_source_node_id);
    }

    pub fn get_sink_node_id(&self) -> u32 {
        self.sink_node_id.unwrap()
    }

    pub fn set_sink_node_id(&mut self, new_sink_node_id: u32) -> () {
        self.sink_node_id = Some(new_sink_node_id);
    }

    pub fn num_nodes(&self) -> usize {
        self.node_ids.len()
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn successors(&self, node_id: u32) -> Option<impl Iterator<Item = &u32>> {
        self.edges.get(&node_id).map(|succs| succs.keys())
    }

    pub fn predecessors(&self, node_id: u32) -> Option<impl Iterator<Item = &u32>> {
        self.preds.get(&node_id).map(|preds| preds.iter())
    }

    pub fn edges(&self) -> impl Iterator<Item = (&u32, &u32, &LowtimeEdge)> {
        self.edges.iter().flat_map(|(from, inner)| {
            inner.iter().map(move |(to, edge)| (from, to, edge))
        })
    }

    pub fn edges_mut(&mut self) -> impl Iterator<Item = (&u32, &u32, &mut LowtimeEdge)> {
        self.edges.iter_mut().flat_map(|(from, inner)| {
            inner.iter_mut().map(move |(to, edge)| (from, to, edge))
        })
    }

    pub fn get_node_ids(&self) -> &Vec<u32> {
        &self.node_ids
    }

    pub fn add_node_id(&mut self, node_id: u32) -> () {
        assert!(self.node_ids.last().unwrap() < &node_id, "New node ids must be larger than all existing node ids");
        self.node_ids.push(node_id)
    }

    pub fn has_edge(&self, from: u32, to: u32) -> bool {
        match self.edges.get(&from).and_then(|inner| inner.get(&to)) {
            Some(_) => true,
            None => false,
        }
    }

    pub fn get_edge(&self, from: u32, to: u32) -> &LowtimeEdge {
        self.edges.get(&from)
            .and_then(|inner| inner.get(&to))
            .expect(&format!("Edge {} to {} not found", from, to))
    }

    pub fn get_edge_mut(&mut self, from: u32, to: u32) -> &mut LowtimeEdge {
        self.edges.get_mut(&from)
            .and_then(|inner| inner.get_mut(&to))
            .expect(&format!("Edge {} to {} not found", from, to))
    }

    pub fn add_edge(&mut self, from: u32, to: u32, edge: LowtimeEdge) -> () {
        self.edges.entry(from).or_insert_with(HashMap::new).insert(to, edge);
        self.preds.entry(to).or_insert_with(HashSet::new).insert(from);
        self.num_edges += 1;
    }

    // fn get_mut_op(&mut self, from: u32, to: u32) -> Option<&mut LowtimeEdge> {
    //     self.edges
    //         .get_mut(&from)
    //         .and_then(|to_edges| to_edges.get_mut(&to))
    // }

    // TESTING(ohjun): should make private when testing functions are deleted
    pub fn get_ek_preprocessed_edges(&self, ) -> Vec<Edge<u32, OrderedFloat<f64>>> {
        let mut processed_edges = Vec::with_capacity(self.num_edges);
        processed_edges.extend(
            self.edges.iter().flat_map(|(from, inner)|
            inner.iter().map(|(to, edge)|
                ((*from, *to), edge.get_capacity())
        )));
        processed_edges
    }

    // TESTING(ohjun)
    pub fn print_all_capacities(&self) -> () {
        let mut processed_edges = self.get_ek_preprocessed_edges();
        processed_edges.sort_by(|((a_from, a_to), _a_cap): &((u32, u32), OrderedFloat<f64>),
                                 ((b_from, b_to), _b_cap): &((u32, u32), OrderedFloat<f64>)| {
            // a_from < b_from || (a_from == b_from && a_to < b_to)
            let from_cmp = a_from.cmp(&b_from);
            if from_cmp == Ordering::Equal {
                a_to.cmp(&b_to)
            }
            else {
                from_cmp
            }
        });
        info!("Rust Printing graph:");
        info!("Num edges: {}", processed_edges.len());
        processed_edges.iter().for_each(|((from, to), cap)| {
            info!("{} -> {}: {}", from, to, cap);
        });
    }
}

#[derive(Clone)]
pub struct LowtimeEdge {
    op: Option<Operation>,
    capacity: OrderedFloat<f64>,
    flow: OrderedFloat<f64>,
    ub: OrderedFloat<f64>,
    lb: OrderedFloat<f64>,
}

impl LowtimeEdge {
    pub fn new(
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

    // TODO(ohjun): there's probably a better way to do this with default args
    pub fn new_only_capacity(capacity: OrderedFloat<f64>) -> Self {
        LowtimeEdge {
            op: None,
            capacity,
            flow: OrderedFloat(0.0),
            ub: OrderedFloat(0.0),
            lb: OrderedFloat(0.0),
        }
    }

    // TODO(ohjun): there's probably a better way to do this with default args
    pub fn new_only_flow(flow: OrderedFloat<f64>) -> Self {
        LowtimeEdge {
            op: None,
            capacity: OrderedFloat(0.0),
            flow,
            ub: OrderedFloat(0.0),
            lb: OrderedFloat(0.0),
        }
    }

    pub fn get_capacity(&self) -> OrderedFloat<f64> {
        self.capacity
    }

    pub fn set_capacity(&mut self, new_capacity: OrderedFloat<f64>) -> () {
        self.capacity = new_capacity
    }

    pub fn get_flow(&self) -> OrderedFloat<f64> {
        self.flow
    }

    pub fn set_flow(&mut self, new_flow: OrderedFloat<f64>) -> () {
        self.flow = new_flow;
    }

    pub fn incr_flow(&mut self, flow: OrderedFloat<f64>) -> () {
        self.flow += flow;
    }

    pub fn decr_flow(&mut self, flow: OrderedFloat<f64>) -> () {
        self.flow -= flow;
    }

    pub fn get_ub(&self) -> OrderedFloat<f64> {
        self.ub
    }

    pub fn get_lb(&self) -> OrderedFloat<f64> {
        self.lb
    }

}