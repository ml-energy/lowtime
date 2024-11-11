use ordered_float::OrderedFloat;


pub trait CostModel {
    fn get_cost(&self, duration: u32) -> f64;
}

pub struct Operation<C: CostModel> {
    capacity: OrderedFloat<f64>,
    flow: OrderedFloat<f64>,
    ub: OrderedFloat<f64>,
    lb: OrderedFloat<f64>,
    // TODO(ohjun): should we have a separate struct for node values above and operation values?
    duration: u32,
    cost_model: C,
}

impl<C> Operation<C> 
where
    C: CostModel,
{
    fn new(capacity: f64, flow: f64, ub: f64, lb: f64, duration: u32, cost_model: C) -> Self {
        Operation {
            capacity: OrderedFloat(capacity),
            flow: OrderedFloat(flow),
            ub: OrderedFloat(ub),
            lb: OrderedFloat(lb),
            duration,
            cost_model,
        }
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

    fn get_duration(&self) -> u32 {
        self.duration
    }

    fn get_cost(&self, duration: u32) -> f64 {
        self.cost_model.get_cost(duration)
    }
}

struct ExponentialModel {
    a: f64,
    b: f64,
    c: f64,
    // TODO(ohjun): consider impact of manual caching
    // cache: HashMap<u32, f64>
}

impl ExponentialModel {
    fn new(a: f64, b: f64, c: f64) -> Self {
        ExponentialModel {a, b, c}
    }
}

impl CostModel for ExponentialModel {
    fn get_cost(&self, duration: u32) -> f64 {
        self.a * f64::exp(self.b * duration as f64) + self.c
    }
}