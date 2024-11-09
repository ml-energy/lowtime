use pyo3::prelude::*;

mod utils;
mod phillips_dessouky;
mod lowtime_graph;

use phillips_dessouky::PhillipsDessouky;
use lowtime_graph::LowtimeGraph;

#[pymodule]
fn _lowtime_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();  // send Rust logs to Python logger
    m.add_class::<PhillipsDessouky>()?;
    Ok(())
}
