use ordered_float::OrderedFloat;

use crate::cost_model::CostModel;


pub struct Operation {
    capacity: OrderedFloat<f64>,
    flow: OrderedFloat<f64>,
    ub: OrderedFloat<f64>,
    lb: OrderedFloat<f64>,
    duration: u32,
    cost_model: CostModel,
}

impl Operation {
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

    fn get_cost(&mut self, duration: u32) -> f64 {
        self.cost_model.get_cost(duration)
    }
}
