use pyo3::prelude::*;

use std::collections::HashSet;

use ordered_float::OrderedFloat;
use pathfinding::directed::edmonds_karp::{
    SparseCapacity,
    Edge,
    edmonds_karp
};

use std::time::Instant;
use log::{info, debug, Level};

use crate::lowtime_graph::{LowtimeGraph, LowtimeEdge};
use crate::utils;


#[pyclass]
pub struct PhillipsDessouky {
   dag: LowtimeGraph,
   unbound_dag_temp: LowtimeGraph, // TESTING(ohjun)
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
            dag: LowtimeGraph::of_python(
                node_ids,
                source_node_id,
                sink_node_id,
                edges_raw,
            ),
            unbound_dag_temp: LowtimeGraph::new(), // TESTING(ohjun)
        })
    }

    // TESTING ohjun
    fn get_dag_node_ids(&self) -> Vec<u32> {
        self.dag.get_node_ids().clone()
    }

    // TESTING ohjun
    fn get_dag_ek_processed_edges(&self) -> Vec<((u32, u32), f64)> {
        let rs_edges = self.dag.get_ek_preprocessed_edges();
        let py_edges: Vec<((u32, u32), f64)> = rs_edges.iter().map(|((from, to), cap)| {
            ((*from, *to), cap.into_inner())
        }).collect();
        py_edges
    }

    // TESTING ohjun
    fn get_unbound_dag_node_ids(&self) -> Vec<u32> {
        self.unbound_dag_temp.get_node_ids().clone()
    }

    // TESTING ohjun
    fn get_unbound_dag_ek_processed_edges(&self) -> Vec<((u32, u32), f64)> {
        let rs_edges = self.unbound_dag_temp.get_ek_preprocessed_edges();
        let py_edges: Vec<((u32, u32), f64)> = rs_edges.iter().map(|((from, to), cap)| {
            ((*from, *to), cap.into_inner())
        }).collect();
        py_edges
    }

    // TESTING(ohjun)
    fn max_flow_depr(&self) -> Vec<((u32, u32), f64)> {
        info!("CALLING MAX FLOW FROM max_flow_depr");
        let (flows, _, _) = self.dag.max_flow();
        let flows_f64: Vec<((u32, u32), f64)> = flows.iter().map(|((from, to), flow)| {
            ((*from, *to), flow.into_inner())
        }).collect();
        flows_f64
    }

    // TODO(ohjun): iteratively implement/verify in _wip until done,
    //              then replace with this function
    // fn find_min_cut(&self) -> (HashSet, HashSet) {
    // }

    fn find_min_cut_wip(&mut self) -> Vec<((u32, u32), f64)> {      
        // In order to solve max flow on edges with both lower and upper bounds,
        // we first need to convert it to another DAG that only has upper bounds.
        let mut unbound_dag: LowtimeGraph = self.dag.clone();

        // For every edge, capacity = ub - lb.
        unbound_dag.edges_mut().for_each(|(_from, _to, edge)|
            edge.set_capacity(edge.get_ub() - edge.get_lb())
        );

        // Add a new node s', which will become the new source node.
        // We constructed the AOA DAG, so we know that node IDs are integers.
        let s_prime_id = unbound_dag.get_node_ids().last().unwrap() + 1;
        unbound_dag.add_node_id(s_prime_id);

        // For every node u in the original graph, add an edge (s', u) with capacity
        // equal to the sum of all lower bounds of u's parents.
        let orig_node_ids = self.dag.get_node_ids();
        for u in orig_node_ids.iter() {
            let mut capacity = OrderedFloat(0.0);
            if let Some(preds) = unbound_dag.predecessors(*u) {
                capacity = preds.fold(OrderedFloat(0.0), |acc, pred_id| {
                    acc + unbound_dag.get_edge(*pred_id, *u).get_lb()
                });
            }
            unbound_dag.add_edge(s_prime_id, *u, LowtimeEdge::new_only_capacity(capacity));
        }

        // Add a new node t', which will become the new sink node.
        let t_prime_id = s_prime_id + 1;
        unbound_dag.add_node_id(t_prime_id);

        // For every node u in the original graph, add an edge (u, t') with capacity
        // equal to the sum of all lower bounds of u's children.
        for u in orig_node_ids.iter() {
            let mut capacity = OrderedFloat(0.0);
            if let Some(succs) = unbound_dag.successors(*u) {
                capacity = succs.fold(OrderedFloat(0.0), |acc, succ_id| {
                    acc + unbound_dag.get_edge(*u, *succ_id).get_lb()
                });
            }
            unbound_dag.add_edge(*u, t_prime_id, LowtimeEdge::new_only_capacity(capacity));
        }

        if log::log_enabled!(Level::Debug) {
            debug!("Unbound DAG");
            debug!("Number of nodes: {}", unbound_dag.num_nodes());
            debug!("Number of edges: {}", unbound_dag.num_edges());
            let total_capacity = unbound_dag.edges()
                .fold(OrderedFloat(0.0), |acc, (_from, _to, edge)| acc + edge.get_capacity());
            debug!("Sum of capacities: {}", total_capacity);
        }

        // Add an edge from t to s with infinite capacity.
        unbound_dag.add_edge(
            unbound_dag.get_sink_node_id(),
            unbound_dag.get_source_node_id(),
            LowtimeEdge::new_only_capacity(OrderedFloat(f64::INFINITY)),
        );

        // Update source and sink on unbound_dag
        unbound_dag.set_source_node_id(s_prime_id);
        unbound_dag.set_sink_node_id(t_prime_id);

        // First Max Flow
        info!("CALLING MAX FLOW FROM find_min_cut_wip"); // TESTING(ohjun)
        let (flows, _, _) = unbound_dag.max_flow();

        if log::log_enabled!(Level::Debug) {
            debug!("After first max flow");
            let total_flow = flows.iter()
                .fold(OrderedFloat(0.0), |acc, ((_from, _to), flow)| acc + flow);
            debug!("Sum of all flow values: {}", total_flow);
        }

        
        let flows_f64: Vec<((u32, u32), f64)> = flows.iter().map(|((from, to), flow)| {
            ((*from, *to), flow.into_inner())
        }).collect();

        //// TESTING(ohjun)
        self.unbound_dag_temp = unbound_dag;

        flows_f64
    }
}

// not exposed to Python
impl PhillipsDessouky {

    // fn max_flow(&self) -> Vec<((u32, u32), f64)> {
    //     self.graph.max_flow()
    // }
}
