use pyo3::prelude::*;

use std::collections::{HashMap, HashSet, VecDeque};
use log::{debug, error, Level};
use ordered_float::OrderedFloat;

use crate::lowtime_graph::{LowtimeGraph, LowtimeEdge};


#[pyclass]
pub struct PhillipsDessouky {
   dag: LowtimeGraph,
   fp_error: f64,
}

#[pymethods]
impl PhillipsDessouky {
    #[new]
    fn new(
        fp_error: f64,
        node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges_raw: Vec<((u32, u32), (f64, f64, f64, f64))>,
    ) -> PyResult<Self> {
        Ok(PhillipsDessouky {
            dag: LowtimeGraph::of_python(
                node_ids,
                source_node_id,
                sink_node_id,
                edges_raw,
            ),
            fp_error,
        })
    }

    /// Find the min cut of the DAG annotated with lower/upper bound flow capacities.
    ///
    /// Assumptions:
    ///     - The capacity DAG is in AOA form.
    ///     - The capacity DAG has been annotated with `lb` and `ub` attributes on edges,
    ///         representing the lower and upper bounds of the flow on the edge.
    ///
    /// Returns:
    ///     A tuple of (s_set, t_set) where s_set is the set of nodes on the source side
    ///     of the min cut and t_set is the set of nodes on the sink side of the min cut.
    /// 
    /// Panics if:
    ///     - Any of the assumptions are not true.
    ///     - No feasible flow exists.
    fn find_min_cut(&mut self) -> (HashSet<u32>, HashSet<u32>) {
        // In order to solve max flow on edges with both lower and upper bounds,
        // we first need to convert it to another DAG that only has upper bounds.
        let mut unbound_dag: LowtimeGraph = self.dag.clone();

        // For every edge, capacity = ub - lb.
        unbound_dag.edges_mut().for_each(|(_from, _to, edge)|
            edge.capacity = OrderedFloat(edge.ub - edge.lb)
        );

        // Add a new node s', which will become the new source node.
        // We constructed the AOA DAG, so we know that node IDs are integers.
        let s_prime_id = unbound_dag.node_ids.last().unwrap() + 1;
        unbound_dag.add_node_id(s_prime_id);

        // For every node u in the original graph, add an edge (s', u) with capacity
        // equal to the sum of all lower bounds of u's parents.
        let orig_node_ids = &self.dag.node_ids;
        for u in orig_node_ids.iter() {
            let mut capacity = OrderedFloat(0.0);
            if let Some(preds) = unbound_dag.predecessors(*u) {
                capacity = OrderedFloat(preds.fold(0.0, |acc, pred_id| {
                    acc + unbound_dag.get_edge(*pred_id, *u).lb
                }));
            }
            unbound_dag.add_edge(s_prime_id, *u, LowtimeEdge::new(
                capacity,
                0.0, // flow
                0.0, // ub
                0.0, // lb
            ));
        }

        // Add a new node t', which will become the new sink node.
        let t_prime_id = s_prime_id + 1;
        unbound_dag.add_node_id(t_prime_id);

        // For every node u in the original graph, add an edge (u, t') with capacity
        // equal to the sum of all lower bounds of u's children.
        for u in orig_node_ids.iter() {
            let mut capacity = OrderedFloat(0.0);
            if let Some(succs) = unbound_dag.successors(*u) {
                capacity = OrderedFloat(succs.fold(0.0, |acc, succ_id| {
                    acc + unbound_dag.get_edge(*u, *succ_id).lb
                }));
            }
            unbound_dag.add_edge(*u, t_prime_id, LowtimeEdge::new(
                capacity,
                0.0, // flow
                0.0, // ub
                0.0, // lb
            ));
        }

        if log::log_enabled!(Level::Debug) {
            debug!("Unbound DAG");
            debug!("Number of nodes: {}", unbound_dag.num_nodes());
            debug!("Number of edges: {}", unbound_dag.num_edges());
            let total_capacity = unbound_dag.edges()
                .fold(OrderedFloat(0.0), |acc, (_from, _to, edge)| acc + edge.capacity);
            debug!("Sum of capacities: {}", total_capacity);
        }

        // Add an edge from t to s with infinite capacity.
        unbound_dag.add_edge(
            unbound_dag.sink_node_id,
            unbound_dag.source_node_id,
            LowtimeEdge::new(
                OrderedFloat(f64::INFINITY), // capacity
                0.0, // flow
                0.0, // ub
                0.0, // lb
            ),
        );

        // Update source and sink on unbound_dag
        unbound_dag.source_node_id = s_prime_id;
        unbound_dag.sink_node_id = t_prime_id;

        // We're done with constructing the DAG with only flow upper bounds.
        // Find the maximum flow on this DAG.
        let (flows, _max_flow, _min_cut): (
            Vec<((u32, u32), OrderedFloat<f64>)>,
            OrderedFloat<f64>,
            Vec<((u32, u32), OrderedFloat<f64>)>,
        ) = unbound_dag.max_flow();

        if log::log_enabled!(Level::Debug) {
            debug!("After first max flow");
            let total_flow = flows.iter()
                .fold(OrderedFloat(0.0), |acc, ((_from, _to), flow)| acc + flow);
            debug!("Sum of all flow values: {}", total_flow);
        }

        // Convert flows to dict for faster lookup
        let flow_dict = flows.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<u32, HashMap<u32, f64>>, ((from, to), flow)| {
                acc.entry(*from)
                    .or_insert_with(HashMap::new)
                    .insert(*to, flow.into_inner());
                acc
            }
        );

        // Check if residual graph is saturated. If so, we have a feasible flow.
        if let Some(succs) = unbound_dag.successors(s_prime_id) {
            for u in succs {
                let flow = flow_dict.get(&s_prime_id)
                    .and_then(|inner| inner.get(u))
                    .unwrap_or(&0.0);
                let cap = unbound_dag.get_edge(s_prime_id, *u).capacity;
                let diff = (flow - cap.into_inner()).abs();
                if diff > self.fp_error {
                    error!(
                        "s' -> {} unsaturated (flow: {}, capacity: {})",
                        u,
                        flow_dict[&s_prime_id][u],
                        unbound_dag.get_edge(s_prime_id, *u).capacity,
                    );
                    panic!("ERROR: Max flow on unbounded DAG didn't saturate.");
                }
            }
        }
        if let Some(preds) = unbound_dag.predecessors(t_prime_id) {
            for u in preds {
                let flow = flow_dict.get(u)
                    .and_then(|inner| inner.get(&t_prime_id))
                    .unwrap_or(&0.0);
                let cap = unbound_dag.get_edge(*u, t_prime_id).capacity;
                let diff = (flow - cap.into_inner()).abs();
                if diff > self.fp_error {
                    error!(
                        "{} -> t' unsaturated (flow: {}, capacity: {})",
                        u,
                        flow_dict[u][&t_prime_id],
                        unbound_dag.get_edge(*u, t_prime_id).capacity,
                    );
                    panic!("ERROR: Max flow on unbounded DAG didn't saturate.");
                }
            }
        }

        // We have a feasible flow. Construct a new residual graph with the same
        // shape as the capacity DAG so that we can find the min cut.
        // First, retrieve the flow amounts to the original capacity graph, where for
        // each edge u -> v, the flow amount is `flow + lb`.
        for (u, v, edge) in self.dag.edges_mut() {
            let flow = flow_dict.get(u)
                .and_then(|inner| inner.get(v))
                .unwrap_or(&0.0);
            edge.flow = flow + edge.lb;
        }

        // Construct a new residual graph (same shape as capacity DAG) with
        // u -> v capacity `ub - flow` and v -> u capacity `flow - lb`.
        let mut residual_graph = self.dag.clone();
        for (u, v, _dag_edge) in self.dag.edges() {
            // Rounding small negative values to 0.0 avoids pathfinding::edmonds_karp
            // from entering unreachable code. Has no impact on correctness in test runs.
            let residual_uv_edge = residual_graph.get_edge_mut(*u, *v);
            let mut uv_capacity = OrderedFloat(residual_uv_edge.ub - residual_uv_edge.flow);
            if uv_capacity.into_inner().abs() < self.fp_error {
                uv_capacity = OrderedFloat(0.0);
            }
            residual_uv_edge.capacity = uv_capacity;

            let mut vu_capacity = OrderedFloat(residual_uv_edge.flow - residual_uv_edge.lb);
            if vu_capacity.into_inner().abs() < self.fp_error {
                vu_capacity = OrderedFloat(0.0);
            }

            match self.dag.has_edge(*v, *u) {
                true => residual_graph.get_edge_mut(*v, *u).capacity = vu_capacity,
                false => residual_graph.add_edge(*v, *u, LowtimeEdge::new(
                    vu_capacity,
                    0.0, // flow
                    0.0, // ub
                    0.0, // lb
                )),
            }
        }

        // Run max flow on the new residual graph.
        let (flows, _max_flow, _min_cut): (
            Vec<((u32, u32), OrderedFloat<f64>)>,
            OrderedFloat<f64>,
            Vec<((u32, u32), OrderedFloat<f64>)>,
        ) = residual_graph.max_flow();

        // Convert flows to dict for faster lookup
        let flow_dict = flows.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<u32, HashMap<u32, f64>>, ((from, to), flow)| {
                acc.entry(*from)
                    .or_insert_with(HashMap::new)
                    .insert(*to, flow.into_inner());
                acc
            }
        ); 

        // Add additional flow we get to the original graph
        for (u, v, edge) in self.dag.edges_mut() {
            edge.flow += *flow_dict.get(u)
                .and_then(|inner| inner.get(v))
                .unwrap_or(&0.0);
            edge.flow -= *flow_dict.get(v)
                .and_then(|inner| inner.get(u))
                .unwrap_or(&0.0);
        }

        // Construct the new residual graph.
        let mut new_residual = self.dag.clone();
        for (u, v, edge) in self.dag.edges() {
            new_residual.get_edge_mut(*u, *v).flow = edge.ub - edge.flow;
            new_residual.add_edge(*v, *u, LowtimeEdge::new(
                OrderedFloat(0.0), // capacity
                edge.flow - edge.lb, // flow
                0.0, // ub
                0.0, // lb
            ));
        }

        if log::log_enabled!(Level::Debug) {
            debug!("New residual graph");
            debug!("Number of nodes: {}", new_residual.num_nodes());
            debug!("Number of edges: {}", new_residual.num_edges());
            let total_flow = unbound_dag.edges()
                .fold(0.0, |acc, (_from, _to, edge)| acc + edge.flow);
            debug!("Sum of capacities: {}", total_flow);
        }

        // Find the s-t cut induced by the second maximum flow above.
        // Only following `flow > 0` edges, find the set of nodes reachable from
        // source node. That's the s-set, and the rest is the t-set.
        let mut s_set: HashSet<u32> = HashSet::new();
        let mut q: VecDeque<u32> = VecDeque::new();
        q.push_back(new_residual.source_node_id);
        while let Some(cur_id) = q.pop_back() {
            s_set.insert(cur_id);
            if cur_id == new_residual.sink_node_id {
                break;
            }
            if let Some(succs) = new_residual.successors(cur_id) {
                for child_id in succs {
                    let flow = new_residual.get_edge(cur_id, *child_id).flow;
                    if !s_set.contains(child_id) && flow.abs() > self.fp_error {
                        q.push_back(*child_id);
                    }
                }
            }
        }
        let all_nodes: HashSet<u32> = new_residual.node_ids.iter().copied().collect();
        let t_set: HashSet<u32> = all_nodes.difference(&s_set).copied().collect();
        (s_set, t_set)
    }
}
