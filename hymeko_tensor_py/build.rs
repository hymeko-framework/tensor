//! Link arguments for the Python extension, kept INSIDE this crate.
//!
//! A CPython extension does not link `libpython`: the interpreter already holds those symbols and supplies them
//! when it loads the module. Apple's linker rejects undefined symbols by default, so the extension needs
//! `-undefined dynamic_lookup`.
//!
//! The usual place for that is `.cargo/config.toml` — which is CORE here (CORE.YAML lists it, and
//! `hymeko_spline/tests/build_contract.rs` fails if its `+sse4.1` contract is disturbed). Emitting the flag from
//! this crate's own build script keeps a Python concern out of a file every crate in the workspace compiles under.
//!
//! `rustc-link-arg-cdylib` applies to the cdylib target only, so the `rlib` used by `cargo test` is untouched.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
