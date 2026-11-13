//! Python surface of the RL bridge layouts: descriptors as numpy arrays.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::descriptor::{flags, CHOICES, ENT_KINDS, FAMILIES, FIELDS, PRICE_KINDS};

/// The global descriptor vocabularies (title-independent, append-only):
/// `fields` (the descriptor columns), `families` (`family` ids),
/// `map_families` (family ids the policy head factors over the map),
/// `ent_kinds`, `price_kinds` and their `price_cell_types` (the
/// `price_pmf` cell layout, None where not fixed yet), `choices`, `flags`
/// ({name: bit}) and `none` (the "no value" sentinel, -1).
#[pyfunction]
pub fn rl_descriptor_vocab(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("fields", FIELDS.to_vec())?;
    d.set_item("families", FAMILIES.iter().map(|f| f.name()).collect::<Vec<_>>())?;
    d.set_item(
        "map_families",
        FAMILIES
            .iter()
            .filter(|f| f.is_map_family())
            .map(|f| f.id())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("ent_kinds", ENT_KINDS.iter().map(|k| k.name()).collect::<Vec<_>>())?;
    d.set_item("price_kinds", PRICE_KINDS.iter().map(|k| k.name()).collect::<Vec<_>>())?;
    d.set_item(
        "price_cell_types",
        PRICE_KINDS.iter().map(|k| k.cell_type()).collect::<Vec<_>>(),
    )?;
    d.set_item("choices", CHOICES.to_vec())?;
    let f = PyDict::new(py);
    for (name, bit) in flags::ALL {
        f.set_item(name, bit)?;
    }
    d.set_item("flags", f)?;
    d.set_item("none", super::descriptor::NONE)?;
    Ok(d)
}

/// A title's RL action layout: `title`, `version`, `kind` ("factored": 1830's
/// frozen layout over `factored_legal_indices`; "generator": a layout over the
/// title's legal-move generator, `BaseGame.rl_legal_indices`), `total`,
/// `fingerprint` (16 hex digits), `descriptors` (int32 numpy array
/// `[total, len(fields)]`, columns in `fields` order), `blocks` (list of
/// dicts: name, family, family_id, offset, size, price_kind, flags, axes =
/// [(axis name, [entries])]), `vocab` (the title's corporation / company /
/// train / hex / tile lists the descriptor fields index), plus the global
/// vocabularies of [`rl_descriptor_vocab`]. RuntimeError "not bridge-ready"
/// for a title without a layout.
#[pyfunction]
pub fn rl_layout_info<'py>(py: Python<'py>, title: &str) -> PyResult<Bound<'py, PyDict>> {
    let lo = super::rl_layout(title)?;
    let d = rl_descriptor_vocab(py)?;
    d.set_item("title", lo.title)?;
    d.set_item("version", lo.version.as_str())?;
    d.set_item("kind", crate::title::resolve(lo.title).rl_layout_kind().name())?;
    d.set_item("total", lo.total)?;
    d.set_item("fingerprint", lo.fingerprint_hex())?;
    let flat: Vec<i32> = lo.descriptors.iter().flat_map(|x| x.to_row()).collect();
    let arr = PyArray1::from_vec(py, flat).reshape([lo.total as usize, FIELDS.len()])?;
    d.set_item("descriptors", arr)?;
    let blocks = PyList::empty(py);
    for b in &lo.blocks {
        let bd = PyDict::new(py);
        bd.set_item("name", b.spec.name.as_str())?;
        bd.set_item("family", b.spec.family.name())?;
        bd.set_item("family_id", b.spec.family.id())?;
        bd.set_item("offset", b.offset)?;
        bd.set_item("size", b.size)?;
        bd.set_item("price_kind", b.spec.price.map(|k| k.name()))?;
        bd.set_item("flags", b.spec.flags)?;
        let axes: Vec<(String, Vec<String>)> = b.spec.axes.iter().map(|a| (a.name(), a.entries())).collect();
        bd.set_item("axes", axes)?;
        blocks.append(bd)?;
    }
    d.set_item("blocks", blocks)?;
    let v = PyDict::new(py);
    v.set_item("corporation", lo.vocab.corporations.clone())?;
    v.set_item("company", lo.vocab.companies.clone())?;
    v.set_item("train", lo.vocab.trains.clone())?;
    v.set_item("hex", lo.vocab.hexes.clone())?;
    v.set_item("tile", lo.vocab.tiles.clone())?;
    d.set_item("vocab", v)?;
    Ok(d)
}

/// A title's RL readiness: `layout_ready`, `layout_kind` ("none" /
/// "factored" / "generator"), `encoder_ready` (the legacy flat encoder).
/// ValueError for an unknown title.
#[pyfunction]
pub fn rl_title_status<'py>(py: Python<'py>, title: &str) -> PyResult<Bound<'py, PyDict>> {
    let Some(t) = crate::title::all_titles().iter().find(|t| t.name() == title) else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown game title: {title}"
        )));
    };
    let d = PyDict::new(py);
    d.set_item("title", t.name())?;
    d.set_item("layout_ready", t.rl_layout_ready())?;
    d.set_item("layout_kind", t.rl_layout_kind().name())?;
    d.set_item("encoder_ready", t.rl_encoder_ready())?;
    Ok(d)
}

/// Register the module's functions (one line in `lib.rs`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rl_descriptor_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(rl_layout_info, m)?)?;
    m.add_function(wrap_pyfunction!(rl_title_status, m)?)?;
    super::encode::py::register(m)?;
    Ok(())
}
