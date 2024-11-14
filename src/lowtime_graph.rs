use std::collections::HashMap;

use std::time::Instant;
use log::info;

use crate::operation::Operation;
use crate::utils;


pub struct LowtimeGraph {
    edges: HashMap<u32, HashMap<u32, Operation>>,
    preds: HashMap<u32, u32>,
}

impl LowtimeGraph {
    pub fn new() -> Self {
        LowtimeGraph {
            edges: HashMap::new(),
            preds: HashMap::new(),
        }
    }

    pub fn of_python(
        node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges: Vec<((u32, u32), f64)>,
    ) -> Self {
        let graph = LowtimeGraph::new();
        // edges.iter().map(|((from, to), capacity)| {
        //     let cost_model = C::new(0, 0, 0);
        //     let op = Operation::new(capacity, 0.0, 0.0, 0.0, 0, cost_model);
        //     graph.add_edge(from, to, op)
        // });
        graph
    }

    pub fn get_node_ids(&self) -> Vec<u32> {
        let mut node_ids: Vec<u32> = self.edges.keys().cloned().collect();
        node_ids.sort();
        node_ids
    }

    pub fn add_edge(&mut self, from: u32, to: u32, op: Operation) -> () {
        self.edges
            .entry(from)
            .or_insert_with(HashMap::new)
            .insert(to, op);
        self.preds.insert(to, from);
    }

    pub fn get_op(&self, from: u32, to: u32) -> Option<&Operation> {
        self.edges
            .get(&from)
            .and_then(|ops| ops.get(&to))
    }

    pub fn get_mut_op(&mut self, from: u32, to: u32) -> Option<&mut Operation> {
        self.edges
            .get_mut(&from)
            .and_then(|ops| ops.get_mut(&to))
    }
}
