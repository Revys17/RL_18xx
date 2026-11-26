//! The site layer (docs/SITE_ENGINE_RS.md): the site's game JSON in, Ruby's undo filter, views, menus, the shadow
//! key, the `site_engine` CLI and the wasm-bindgen package. Only `records` (the storage behind
//! `BaseGame::site`) and `rules` (the helpers of the `site_rules` branches) are compiled without the `site` feature;
//! everything the engine does differently in site mode is behind the runtime flag `BaseGame::site_rules`.

pub mod records;
