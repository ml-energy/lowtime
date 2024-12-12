use std::collections::{HashMap, HashSet};
use ordered_float::OrderedFloat;
use pathfinding::directed::edmonds_karp::{
    SparseCapacity,
    Edge,
    EKFlows,
    edmonds_karp,
};


#[derive(Clone)]
pub struct LowtimeGraph {
    pub node_ids: Vec<u32>,
    pub source_node_id: u32,
    pub sink_node_id: u32,
    edges: HashMap<u32, HashMap<u32, LowtimeEdge>>,
    preds: HashMap<u32, HashSet<u32>>,
    num_edges: usize,
}

impl LowtimeGraph {
    pub fn new(source_node_id: u32, sink_node_id: u32) -> Self {
        LowtimeGraph {
            node_ids: Vec::new(),
            source_node_id,
            sink_node_id,
            edges: HashMap::new(),
            preds: HashMap::new(),
            num_edges: 0,
        }
    }

    pub fn of_python(
        mut node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges_raw: Vec<((u32, u32), (f64, f64, f64, f64))>,
    ) -> Self {
        let mut graph = LowtimeGraph::new(source_node_id, sink_node_id);
        node_ids.sort();
        graph.node_ids = node_ids.clone();

        edges_raw.iter().for_each(|(
                (from, to),
                (capacity, flow, ub, lb),
            )| {
            graph.add_edge(*from, *to, LowtimeEdge::new(
                OrderedFloat(*capacity),
                *flow,
                *ub,
                *lb,
            ))
        });
        graph
    }

    pub fn max_flow(&self) -> EKFlows<u32, OrderedFloat<f64>> {
        let edges_edmonds_karp = self.get_ek_preprocessed_edges();
        edmonds_karp::<_, _, _, SparseCapacity<_>>(
            &self.node_ids,
            &self.source_node_id,
            &self.sink_node_id,
            edges_edmonds_karp,
        )
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

    fn get_ek_preprocessed_edges(&self, ) -> Vec<Edge<u32, OrderedFloat<f64>>> {
        let mut processed_edges = Vec::with_capacity(self.num_edges);
        processed_edges.extend(
            self.edges.iter().flat_map(|(from, inner)|
            inner.iter().map(|(to, edge)|
                ((*from, *to), edge.capacity)
        )));
        processed_edges
    }
}

#[derive(Clone)]
pub struct LowtimeEdge {
    pub capacity: OrderedFloat<f64>,
    pub flow: f64,
    pub ub: f64,
    pub lb: f64,
}

impl LowtimeEdge {
    pub fn new(
        capacity: OrderedFloat<f64>,
        flow: f64,
        ub: f64,
        lb: f64,
    ) -> Self {
        LowtimeEdge {
            capacity,
            flow,
            ub,
            lb,
        }
    }
}
