use pyo3::prelude::*;

mod lowtime_graph;
mod phillips_dessouky;
mod utils;

use phillips_dessouky::PhillipsDessouky;


#[pymodule]
fn _lowtime_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();  // send Rust logs to Python logger
    m.add_class::<PhillipsDessouky>()?;
    Ok(())
}
