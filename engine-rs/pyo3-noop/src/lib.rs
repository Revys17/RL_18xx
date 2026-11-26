//! No-op stand-ins for pyo3's attribute macros in builds without Python: they drop the
//! pyo3 helper attributes (`#[pyo3(..)]`, `#[getter]`, `#[new]`, ...) and leave plain Rust.
use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

const HELPERS: &[&str] = &["pyo3", "getter", "setter", "new", "staticmethod", "classmethod", "classattr", "args", "text_signature"];

fn strip(ts: TokenStream) -> TokenStream {
    let toks: Vec<TokenTree> = ts.into_iter().collect();
    let mut out = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        if let (TokenTree::Punct(p), Some(TokenTree::Group(g))) = (&toks[i], toks.get(i + 1)) {
            if p.as_char() == '#' && g.delimiter() == Delimiter::Bracket {
                if let Some(TokenTree::Ident(id)) = g.stream().into_iter().next() {
                    if HELPERS.contains(&id.to_string().as_str()) {
                        i += 2;
                        continue;
                    }
                }
            }
        }
        match &toks[i] {
            TokenTree::Group(g) => {
                let mut ng = Group::new(g.delimiter(), strip(g.stream()));
                ng.set_span(g.span());
                out.push(TokenTree::Group(ng));
            }
            t => out.push(t.clone()),
        }
        i += 1;
    }
    out.into_iter().collect()
}

#[proc_macro_attribute]
pub fn pyclass(_a: TokenStream, item: TokenStream) -> TokenStream { strip(item) }
#[proc_macro_attribute]
pub fn pymethods(_a: TokenStream, item: TokenStream) -> TokenStream { strip(item) }
