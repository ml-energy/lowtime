use std::collections::HashMap;

use std::time::Instant;
use log::info;

use crate::operation::{Operation, CostModel};
use crate::utils;


pub struct LowtimeGraph<C: CostModel> {
    edges: HashMap<u32, HashMap<u32, Operation<C>>>,
    preds: HashMap<u32, u32>,
}

impl<C> LowtimeGraph<C>
where
    C: CostModel
{
    pub fn new() -> Self {
        LowtimeGraph {
            edges: HashMap::new(),
            preds: HashMap::new(),
        }
    }

    pub fn get_node_ids(&self) -> Vec<u32> {
        let mut node_ids: Vec<u32> = self.edges.keys().cloned().collect();
        node_ids.sort();
        node_ids
    }

    pub fn add_edge(&mut self, from: u32, to: u32, op: Operation<C>) -> () {
        self.edges
            .entry(from)
            .or_insert_with(HashMap::new)
            .insert(to, op);
        self.preds.insert(to, from);
    }

    pub fn get_op(&self, from: u32, to: u32) -> Option<&Operation<C>> {
        self.edges
            .get(&from)
            .and_then(|ops| ops.get(&to))
    }

    pub fn get_mut_op(&mut self, from: u32, to: u32) -> Option<&mut Operation<C>> {
        self.edges
            .get_mut(&from)
            .and_then(|ops| ops.get_mut(&to))
    }
}
