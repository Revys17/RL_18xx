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
#[cfg(feature = "site")]
pub mod acts;
#[cfg(feature = "site")]
pub mod g1867;
#[cfg(feature = "site")]
pub mod authority;
#[cfg(feature = "site")]
pub mod check1830;
#[cfg(feature = "site")]
pub mod route_check;
#[cfg(feature = "site")]
pub mod programs;
#[cfg(feature = "wasm")]
pub mod wasm;

/// The titles the site layer plays (docs/SITE_ENGINE_RS_1867.md §3.1): what `version()` lists.
pub const TITLES: &[&str] = &["1830", "1867"];

/// The engine's build: what `WasmGame.version()` and `site_engine version` print (docs/SITE_ENGINE_RS.md §3.2; API 3:
/// docs/SITE_ENGINE_RS_STEP3.md §10, K1').
pub const API_VERSION: u32 = 3;
/// The shadow key's version (docs/SITE_ENGINE_RS.md §5.3; v2: docs/SITE_ENGINE_RS_1867.md §6).
pub const KEY_VERSION: u32 = 2;
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
#[cfg(all(test, feature = "site"))]
mod tests_authority;
