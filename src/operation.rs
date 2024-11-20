// use crate::cost_model::CostModel;


#[derive(Clone)]
pub struct Operation {
    is_dummy: bool,

    duration: u64,
    max_duration: u64,
    min_duration: u64,

    earliest_start: u64,
    latest_start: u64,
    earliest_finish: u64,
    latest_finish: u64,

    // cost_model: CostModel,
}

impl Operation {
    pub fn new(
        is_dummy: bool,
        duration: u64,
        max_duration: u64,
        min_duration: u64, 
        earliest_start: u64,
        latest_start: u64,
        earliest_finish: u64,
        latest_finish: u64,
        // cost_model: CostModel,
    ) -> Self {
        Operation {
            is_dummy,
            duration,
            max_duration,
            min_duration,
            earliest_start,
            latest_start,
            earliest_finish,
            latest_finish,
            // cost_model,
        }
    }

    pub fn get_duration(&self) -> u64 {
        self.duration
    }

    // fn get_cost(&mut self, duration: u32) -> f64 {
    //     self.cost_model.get_cost(duration)
    // }
}
