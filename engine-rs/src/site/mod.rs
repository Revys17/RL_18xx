//! The site layer (docs/SITE_ENGINE_RS.md): the site's game JSON in, Ruby's undo filter, views, menus, the shadow
//! key, the `site_engine` CLI and the wasm-bindgen package. Only `records` (the storage behind
//! `BaseGame::site`) and `rules` (the helpers of the `site_rules` branches) are compiled without the `site` feature;
//! everything the engine does differently in site mode is behind the runtime flag `BaseGame::site_rules`.

pub mod records;
pub mod rules;

#[cfg(feature = "site")]
pub mod json;
#[cfg(feature = "site")]
pub mod game;
#[cfg(feature = "site")]
pub mod view;
#[cfg(feature = "site")]
pub mod track;
#[cfg(feature = "site")]
pub mod menu;
#[cfg(feature = "site")]
pub mod details;
#[cfg(feature = "site")]
pub mod log;
#[cfg(feature = "site")]
pub mod shadow;
#[cfg(feature = "site")]
pub mod cli;
#[cfg(feature = "site")]
pub mod compare;
#[cfg(feature = "wasm")]
pub mod wasm;

/// The engine's build: what `WasmGame.version()` and `site_engine version` print (docs/SITE_ENGINE_RS.md §3.2).
pub const API_VERSION: u32 = 2;
/// The shadow key's version (§5.3).
pub const KEY_VERSION: u32 = 1;
/// The commit the engine was built from (site/build-site.sh sets ENGINE_RS_COMMIT; "dev" otherwise).
pub const COMMIT: &str = match option_env!("ENGINE_RS_COMMIT") {
    Some(c) => c,
    None => "dev",
};

/// The engine's legal action types now (debugging: `site_replay`'s SITE_AT).
#[cfg(feature = "site")]
pub fn debug_legal_types(g: &mut game::SiteGame) -> Vec<String> {
    g.game.legal_action_types()
}

/// The engine's upgradeable tiles for a hex (debugging).
#[cfg(feature = "site")]
pub fn debug_upgradeable(g: &game::SiteGame, hex: &str) -> (Vec<(String, u8)>, Vec<String>) {
    let mut names: Vec<String> = g.game.tile_counts_remaining.keys().cloned().collect();
    names.sort();
    (g.game.upgradeable_tiles_for(hex), names.into_iter().take(5).collect())
}

#[cfg(all(test, feature = "site"))]
mod tests;
