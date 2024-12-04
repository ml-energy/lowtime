use std::cmp;

use log::{info, debug, Level};

use crate::lowtime_graph::{LowtimeGraph, LowtimeEdge};


/// Convert the AOA DAG to a critical AOA DAG where only critical edges remain.
///
/// This function modifies the earliest/latest start/end times of the `Operation`s
/// on the given graph.
///
/// Assumptions:
///     - The graph is a DAG with `Operation`s annotated on edges.
///     - The graph has only one source node, annotated as "source_node" on the graph.
///     - The graph has only one sink node, annotated as "sink_node" on the graph.
// TODO(ohjun): adapt comment above for Rust version
pub fn aoa_to_critical_dag(mut aoa_dag: LowtimeGraph) -> LowtimeGraph {
    // Note: Python version checked whether aoa_dag is a dag; this Rust version does
    // not. This extra check could be added in the future.

    // Clear all earliest/latest start/end times.
    aoa_dag.edges_mut().for_each(|(_, _, edge)| edge.get_op_mut().reset_times());

    // Run the forward pass to set earliest start/end times.
    for node_id in aoa_dag.get_topological_sorted_node_ids() {
        if let Some(succs) = aoa_dag.successors(node_id) {
            // let succs: Vec<u32> = succs.cloned().collect();
            for succ_id in succs {
                let cur_op = aoa_dag.get_edge(node_id, *succ_id).get_op();
                if let Some(succ_succs) = aoa_dag.successors(*succ_id) {
                    // let succ_succs: Vec<u32> = succ_succs.cloned().collect();
                    for succ_succ_id in succ_succs {
                        {
                        let next_op = aoa_dag.get_edge_mut(*succ_id, *succ_succ_id).get_op_mut();
                        next_op.earliest_start = cmp::max(next_op.earliest_start, cur_op.earliest_finish);
                        next_op.earliest_finish = next_op.earliest_start + next_op.duration;
                        }
                    }
                }
            }
        }
    }
    // for node_id in nx.topological_sort(aoa_dag):
    //     for succ_id in aoa_dag.successors(node_id):
    //         cur_op: Operation = aoa_dag[node_id][succ_id][attr_name]

    //         for succ_succ_id in aoa_dag.successors(succ_id):
    //             next_op: Operation = aoa_dag[succ_id][succ_succ_id][attr_name]

    //             next_op.earliest_start = max(
    //                 next_op.earliest_start,
    //                 cur_op.earliest_finish,
    //             )
    //             next_op.earliest_finish = next_op.earliest_start + next_op.duration

    // # Run the backward pass to set latest start/end times.
    // # For the forward pass, `reset_times` was called on all `Operation`s, so we
    // # didn't have to think about the initial values of earliest/latest_start.
    // # For the backward pass, we need to find the largest `earliest_finish` value
    // # among operations on incoming edges to the sink node, which is when the entire
    // # DAG will finish executing. Then, we set that value as the latest_finish value
    // # for operations on incoming edges to the sink node.
    // sink_node = aoa_dag.graph["sink_node"]
    // dag_earliest_finish = 0
    // for node_id in aoa_dag.predecessors(sink_node):
    //     op: Operation = aoa_dag[node_id][sink_node][attr_name]
    //     dag_earliest_finish = max(dag_earliest_finish, op.earliest_finish)
    // for node_id in aoa_dag.predecessors(sink_node):
    //     op: Operation = aoa_dag[node_id][sink_node][attr_name]
    //     op.latest_finish = dag_earliest_finish
    //     op.latest_start = op.latest_finish - op.duration

    // for node_id in reversed(list(nx.topological_sort(aoa_dag))):
    //     for pred_id in aoa_dag.predecessors(node_id):
    //         cur_op: Operation = aoa_dag[pred_id][node_id][attr_name]

    //         for pred_pred_id in aoa_dag.predecessors(pred_id):
    //             prev_op: Operation = aoa_dag[pred_pred_id][pred_id][attr_name]

    //             prev_op.latest_start = min(
    //                 prev_op.latest_start,
    //                 cur_op.latest_start - prev_op.duration,
    //             )
    //             prev_op.latest_finish = prev_op.latest_start + prev_op.duration

    // # Remove all edges that are not on the critical path.
    // critical_dag = nx.DiGraph(aoa_dag)
    // for u, v, edge_attr in aoa_dag.edges(data=True):
    //     op: Operation = edge_attr[attr_name]
    //     if op.earliest_finish != op.latest_finish:
    //         critical_dag.remove_edge(u, v)

    // # Copy over source and sink node IDs.
    // source_id = critical_dag.graph["source_node"] = aoa_dag.graph["source_node"]
    // sink_id = critical_dag.graph["sink_node"] = aoa_dag.graph["sink_node"]
    // if source_id not in critical_dag and source_id in aoa_dag:
    //     raise RuntimeError(
    //         "Source node was removed from the DAG when getting critical DAG."
    //     )
    // if sink_id not in critical_dag and sink_id in aoa_dag:
    //     raise RuntimeError(
    //         "Sink node was removed from the DAG when getting critical DAG."
    //     )

    // return critical_dag

    return aoa_dag;
}
