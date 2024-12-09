use pyo3::prelude::*;

use std::collections::HashMap;


pub trait ModelCost {
    fn get_cost(&self, duration: u32) -> f64;
}

#[pyclass]
pub struct CostModel {
    inner: Box<dyn ModelCost + Send>,
    cache: HashMap<u32, f64>,
}

#[pymethods]
impl CostModel {
    // Issues with traits of input type
    // #[new]
    // pub fn new(model: Box<dyn ModelCost>) -> PyResult<Self> {
    //     Ok(CostModel {
    //         inner: model,
    //         cache: HashMap::new(),
    //     })
    // }

    // Python functions cannot have generic type parameters
    // #[new]
    // pub fn new<C>(model: C) -> PyResult<Self>
    // where
    //     C: ModelCost + Send
    // {
    //     Ok(CostModel {
    //         inner: Box::new(model),
    //         cache: HashMap::new(),
    //     })
    // }

    // Some error message about f64 not implementing some convoluted trait
    // pub fn new_exponential(a: f64, b: f64, c: f64) -> PyResult<Self> {
    //         Ok(CostModel {
    //             inner: Box::new(ExponentialModel::new(a, b, c)),
    //             cache: HashMap::new(),
    //         })
    // }

    pub fn get_cost(&mut self, duration: u32) -> f64 {
        match self.cache.get(&duration) {
            Some(cost) => *cost,
            None => {
                let cost = self.inner.get_cost(duration);
                self.cache.insert(duration, cost);
                cost
            }
        }
    }
}

struct ExponentialModel {
    a: f64,
    b: f64,
    c: f64,
}

impl ExponentialModel {
    fn new(a: f64, b: f64, c: f64) -> Self {
        ExponentialModel {a, b, c}
    }
}

impl ModelCost for ExponentialModel {
    fn get_cost(&self, duration: u32) -> f64 {
        self.a * f64::exp(self.b * duration as f64) + self.c
    }
}
