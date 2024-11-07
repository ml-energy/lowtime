use pyo3::prelude::*;

use ordered_float::OrderedFloat;

use std::time::Instant;
use log::info;

use crate::utils;


#[pyclass]
pub struct LowtimeGraph {
}

#[pymethods]
impl LowtimeGraph {
    #[new]
    fn new(
    ) -> PyResult<Self> {
        Ok(LowtimeGraph {})
    }
}
