use pyo3::prelude::*;

use std::collections::HashMap;


pub trait ModelCost {
    fn get_cost(&self, duration: u32) -> f64;
}

#[pyclass]
pub struct CostModel {
    cache: HashMap<u32, f64>,
    inner: Box<dyn ModelCost>,
}

#[pymethods]
impl CostModel {
    pub fn new(inner: Box<dyn ModelCost>) -> PyResult<Self> {
        Ok(CostModel { inner, cache: HashMap::new() })
    }

    pub fn get_cost(&mut self, duration: u32) -> f64 {
        // match self.cache.entry(duration) {
        //     HashMap::Entry::Occupied => {
        //         self.cache.get(&duration)?.clone()
        //     }
        //     HashMap::Entry::Empty => {
        //         let cost = self.inner.get_cost(duration);
        //         self.cache.insert(duration, cost);
        //         cost
        //     }
        // }
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
