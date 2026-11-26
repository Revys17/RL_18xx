//! PyResult / PyErr / exceptions facade: pyo3's with the `python` feature, plain Rust otherwise.
#[cfg(feature = "python")]
pub use pyo3::{exceptions, PyErr, PyResult};

#[cfg(not(feature = "python"))]
mod plain {
    #[derive(Debug, Clone)]
    pub struct PyErr(pub String);
    impl std::fmt::Display for PyErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
    }
    pub type PyResult<T> = Result<T, PyErr>;
    pub mod exceptions {
        macro_rules! exc { ($($n:ident),*) => { $(pub struct $n; impl $n { pub fn new_err<A: ToString>(a: A) -> super::PyErr { super::PyErr(format!("{}: {}", stringify!($n), a.to_string())) } })* } }
        exc!(PyValueError, PyRuntimeError, PyIndexError, PyKeyError, PyTypeError, NotImplementedError);
    }
}
#[cfg(not(feature = "python"))]
pub use plain::{exceptions, PyErr, PyResult};
