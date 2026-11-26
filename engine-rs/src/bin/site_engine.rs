//! `site_engine`: engine-rs's site layer for the site's server (docs/SITE_ENGINE_RS.md §5.4, contract K3). One JSON
//! object on stdin, one JSON line on stdout.
//!
//!   site_engine version
//!   site_engine state  < {"game": <site game JSON>, "upto": <action id | null>}
//!   site_engine keys   < {"game": ...}
//!   site_engine act    < {"game": ..., "action": ..., "add_auto_actions": false}
//!
//! Exit 0 whenever a verdict was reached (a Rust refusal included), 2 bad input, 3 unsupported (title / optional
//! rule), 101 a panic.

use std::io::Read;

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let mut input = String::new();
    if cmd != "version" {
        if let Err(e) = std::io::stdin().read_to_string(&mut input) {
            println!("{}", serde_json::json!({"ok": false, "error": format!("bad input: stdin: {e}")}));
            std::process::exit(engine_rs::site::cli::EXIT_BAD_INPUT);
        }
    }
    let (code, out) = engine_rs::site::cli::run(&cmd, &input);
    println!("{out}");
    std::process::exit(code);
}
