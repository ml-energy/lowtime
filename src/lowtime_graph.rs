use pyo3::prelude::*;

use std::collections::HashMap;
use ordered_float::OrderedFloat;

use std::time::Instant;
use log::info;

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

    pub fn get_nodes(&self) -> Vec<u32> {
        let mut nodes: Vec<u32> = self.edges.keys().cloned().collect();
        nodes.sort();
        nodes
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

struct Operation {
    capacity: OrderedFloat<f64>,
    flow: OrderedFloat<f64>,
    ub: OrderedFloat<f64>,
    lb: OrderedFloat<f64>,
}
