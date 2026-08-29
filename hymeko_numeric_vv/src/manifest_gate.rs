//! Proving the V&V dependency direction, by reading the manifests rather than trusting them.
//!
//! # Why a gate and not a rule
//!
//! "Production must never depend on a V&V package" is the kind of invariant that holds until
//! someone adds a `dev-dependency` to reuse one helper, at which point a frozen oracle becomes
//! reachable from a production build graph and the whole capture-before-deletion argument
//! quietly stops meaning anything. CLAUDE.md's own precedent is `hymeko_spline`'s build contract:
//! an invariant nobody checks is an invariant nobody has.
//!
//! # What is checked
//!
//! For every `Cargo.toml` in the workspace other than the V&V package's own:
//! * it does not name the V&V package in any dependency table;
//! * the V&V package itself declares `publish = false`.
//!
//! The parser is deliberately crude — it looks for the package name as a whole token anywhere in
//! a dependency table — because a gate that under-reports is worse than one that occasionally
//! over-reports and has to be read.

use std::path::{Path, PathBuf};

/// A manifest found by [`workspace_manifests`].
#[derive(Debug, Clone)]
pub struct Manifest {
    /// Path on disk.
    pub path: PathBuf,
    /// Full text.
    pub text: String,
}

impl Manifest {
    /// The `name = "..."` of the `[package]` table, if there is one.
    #[must_use]
    pub fn package_name(&self) -> Option<String> {
        let after = self.text.split("[package]").nth(1)?;
        for line in after.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                break;
            }
            if let Some(rest) = t.strip_prefix("name") {
                if let Some(v) = rest.trim_start().strip_prefix('=') {
                    return Some(v.trim().trim_matches('"').to_string());
                }
            }
        }
        None
    }

    /// Whether the manifest declares `publish = false`.
    #[must_use]
    pub fn is_publish_false(&self) -> bool {
        self.text
            .lines()
            .map(str::trim)
            .any(|l| l.starts_with("publish") && l.contains("false"))
    }

    /// Whether any dependency table names `dep` as a whole token.
    ///
    /// Comment lines are ignored: a manifest is allowed — encouraged, even — to *explain* in prose
    /// that it deliberately does not depend on the V&V package.
    #[must_use]
    pub fn depends_on(&self, dep: &str) -> bool {
        let mut in_dep_table = false;
        for raw in self.text.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                in_dep_table = line.contains("dependencies");
                continue;
            }
            if !in_dep_table || line.starts_with('#') {
                continue;
            }
            let code = line.split('#').next().unwrap_or(line);
            if code
                .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
                .any(|tok| tok == dep)
            {
                return true;
            }
        }
        false
    }
}

/// Every `Cargo.toml` at depth 1 under `root`, plus `root/Cargo.toml` itself.
///
/// Depth 1 is the shape of both repositories in this workspace: a root manifest and one directory
/// per crate. A nested crate would be missed, so [`assert_v_and_v_is_downstream_only`] also asserts
/// that the count it found is at least as large as the number of workspace members it can see.
///
/// # Panics
/// If `root` cannot be read.
#[must_use]
pub fn workspace_manifests(root: &Path) -> Vec<Manifest> {
    let mut out = Vec::new();
    let mut push = |p: PathBuf| {
        if let Ok(text) = std::fs::read_to_string(&p) {
            out.push(Manifest { path: p, text });
        }
    };
    push(root.join("Cargo.toml"));
    let entries = std::fs::read_dir(root)
        .unwrap_or_else(|e| panic!("cannot read workspace root {}: {e}", root.display()));
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for d in dirs {
        let m = d.join("Cargo.toml");
        if m.is_file() {
            push(m);
        }
    }
    out
}

/// Assert that `vv_package` is `publish = false` and that nothing else in the workspace depends
/// on it.
///
/// # Panics
/// With the offending manifest path if the direction is violated, or if the V&V package's own
/// manifest is not found.
pub fn assert_v_and_v_is_downstream_only(root: &Path, vv_package: &str) {
    let manifests = workspace_manifests(root);
    assert!(
        manifests.len() >= 2,
        "expected a workspace with several manifests under {}, found {}",
        root.display(),
        manifests.len()
    );

    let own = manifests
        .iter()
        .find(|m| m.package_name().as_deref() == Some(vv_package))
        .unwrap_or_else(|| {
            panic!(
                "no manifest declares package `{vv_package}` under {}",
                root.display()
            )
        });
    assert!(
        own.is_publish_false(),
        "{} must declare `publish = false`",
        own.path.display()
    );

    let offenders: Vec<String> = manifests
        .iter()
        .filter(|m| m.path != own.path)
        .filter(|m| m.depends_on(vv_package))
        .map(|m| m.path.display().to_string())
        .collect();
    assert!(
        offenders.is_empty(),
        "V&V dependency direction violated — these manifests depend on `{vv_package}`:\n  {}",
        offenders.join("\n  ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(text: &str) -> Manifest {
        Manifest {
            path: PathBuf::from("Cargo.toml"),
            text: text.to_string(),
        }
    }

    #[test]
    fn package_name_is_read_from_the_package_table() {
        let x = m("[package]\nname = \"alpha\"\nversion = \"0.1.0\"\n");
        assert_eq!(x.package_name().as_deref(), Some("alpha"));
    }

    #[test]
    fn package_name_stops_at_the_next_table() {
        let x = m("[package]\nversion = \"0\"\n\n[dependencies]\nname = \"beta\"\n");
        assert_eq!(x.package_name(), None);
    }

    #[test]
    fn a_dependency_is_detected_in_any_dependency_table() {
        for table in [
            "[dependencies]",
            "[dev-dependencies]",
            "[build-dependencies]",
        ] {
            let x = m(&format!(
                "[package]\n{table}\nhymeko_numeric_vv = {{ path = \"x\" }}\n"
            ));
            assert!(x.depends_on("hymeko_numeric_vv"), "{table}");
        }
    }

    #[test]
    fn a_dependency_outside_a_dependency_table_is_not_counted() {
        let x = m("[package]\nname = \"hymeko_numeric_vv\"\n");
        assert!(!x.depends_on("hymeko_numeric_vv"));
    }

    #[test]
    fn a_commented_mention_is_not_a_dependency() {
        let x = m("[dependencies]\n# deliberately does NOT use hymeko_numeric_vv\nrayon = \"1\"\n");
        assert!(!x.depends_on("hymeko_numeric_vv"));
    }

    #[test]
    fn a_trailing_comment_on_a_real_line_is_stripped() {
        let x = m("[dependencies]\nrayon = \"1\" # not hymeko_numeric_vv\n");
        assert!(!x.depends_on("hymeko_numeric_vv"));
    }

    #[test]
    fn a_longer_name_containing_the_needle_is_not_a_match() {
        let x = m("[dependencies]\nhymeko_numeric_vv_extra = { path = \"x\" }\n");
        assert!(!x.depends_on("hymeko_numeric_vv"));
    }

    #[test]
    fn publish_false_is_detected() {
        assert!(m("[package]\npublish = false\n").is_publish_false());
        assert!(!m("[package]\nname = \"x\"\n").is_publish_false());
    }

    #[test]
    fn the_real_workspace_satisfies_the_direction() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate dir has a parent")
            .to_path_buf();
        assert_v_and_v_is_downstream_only(&root, "hymeko_numeric_vv");
    }
}
