// use crate::cost_model::CostModel;


#[derive(Clone)]
pub struct Operation {
    is_dummy: bool,
    pub duration: u64,
    pub max_duration: u64,
    pub min_duration: u64,
    pub earliest_start: u64,
    pub latest_start: u64,
    pub earliest_finish: u64,
    pub latest_finish: u64,
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

    pub fn reset_times(&mut self) -> () {
        self.earliest_start = 0;
        self.latest_start = u64::MAX;
        self.earliest_finish = 0;
        self.latest_finish = u64::MAX;
    }
}
