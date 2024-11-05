use pyo3::prelude::*;

mod phillips_dessouky_impl;
use phillips_dessouky_impl::PhillipsDessoukyImpl;


#[pyclass]
struct PhillipsDessouky {
    pimpl: Box<PhillipsDessoukyImpl>,
}

#[pymethods]
impl PhillipsDessouky {
    #[new]
    fn new(
        node_ids: Vec<u32>,
        source_node_id: u32,
        sink_node_id: u32,
        edges_raw: Vec<((u32, u32), f64)>,
    ) -> PyResult<Self> {
        let pimpl = Box::new(
            PhillipsDessoukyImpl::new(
                node_ids,
                source_node_id,
                sink_node_id,
                edges_raw,
            )
        );
        Ok(PhillipsDessouky {pimpl})
    }

    fn max_flow(&self) -> PyResult<Vec<((u32, u32), f64)>> {
        Ok(self.pimpl.max_flow())
    }
}

// A Python module implemented in Rust.
#[pymodule]
fn lowtime_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();  // send Rust logs to Python logging system
    m.add_class::<PhillipsDessouky>()?;
    Ok(())
}
