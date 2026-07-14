mod app;
mod backend;
mod protocol;
mod theme;

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (python_executable, project_root=None, debug=false))]
fn run(
    py: Python<'_>,
    python_executable: String,
    project_root: Option<String>,
    debug: bool,
) -> PyResult<i32> {
    py.detach(move || app::run(&python_executable, project_root.as_deref(), debug))
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pymodule]
fn _alphanus_tui(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run, module)?)?;
    module.add("__version__", "0.2.0")?;
    Ok(())
}
