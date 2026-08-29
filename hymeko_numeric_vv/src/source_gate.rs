//! Scanning production source for tokens that must not survive a retirement.
//!
//! # Why this exists
//!
//! Every retirement in this mission ends with a claim of the form "zero production references to
//! X remain". That claim is only worth something if a test makes it. Grepping by hand at the end
//! of a slice proves the state of one afternoon; a gate proves it on every run afterwards, which
//! is the difference between a retirement and a pause.
//!
//! # Scope discipline
//!
//! A gate must scan *production* only. Test suites, frozen oracles and reports legitimately name
//! the retired thing — that is what preserving provenance means — so a scan that walks everything
//! forbids the very evidence the retirement depends on. [`ProductionScan`] therefore takes the
//! roots explicitly and never guesses.

use std::path::{Path, PathBuf};

/// A recursive scan over a set of production source roots.
#[derive(Debug, Clone)]
pub struct ProductionScan {
    roots: Vec<PathBuf>,
}

/// One `(path, line number, line)` hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    /// File the token was found in.
    pub path: PathBuf,
    /// 1-indexed line number.
    pub line_no: usize,
    /// The full line, trimmed.
    pub line: String,
}

impl ProductionScan {
    /// Scan these roots. Non-existent roots are skipped, so a gate can name a directory that a
    /// later slice deletes without becoming a false failure.
    #[must_use]
    pub fn new<I, P>(roots: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self {
            roots: roots
                .into_iter()
                .map(|p| p.as_ref().to_path_buf())
                .filter(|p| p.exists())
                .collect(),
        }
    }

    /// Every `.rs` file under the roots, sorted, so failure messages are stable.
    #[must_use]
    pub fn files(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for root in &self.roots {
            collect_rs(root, &mut out);
        }
        out.sort();
        out
    }

    /// Lines containing `token`, excluding `//` comment lines.
    ///
    /// Comments are excluded for the same reason the section gates in `nagare` exclude them: a
    /// module's rustdoc legitimately names what it deliberately does not use, and a scan over
    /// prose would forbid the sentence that states the contract. A *call* cannot hide in a `//`
    /// line.
    #[must_use]
    pub fn hits(&self, token: &str) -> Vec<Hit> {
        let mut out = Vec::new();
        for path in self.files() {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (i, raw) in text.lines().enumerate() {
                let line = raw.trim();
                if line.starts_with("//") || line.starts_with("/*") || line.starts_with('*') {
                    continue;
                }
                if line.contains(token) {
                    out.push(Hit {
                        path: path.clone(),
                        line_no: i + 1,
                        line: line.to_string(),
                    });
                }
            }
        }
        out
    }

    /// Assert `token` does not occur in production code.
    ///
    /// # Panics
    /// Listing every hit, so the failure is actionable without a second grep.
    pub fn assert_absent(&self, token: &str) {
        let hits = self.hits(token);
        assert!(
            hits.is_empty(),
            "`{token}` must not appear in production code; {} occurrence(s):\n  {}",
            hits.len(),
            hits.iter()
                .map(|h| format!("{}:{}  {}", h.path.display(), h.line_no, h.line))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    /// Assert the scan actually looked at something.
    ///
    /// A gate over an empty file set passes vacuously; this is the guard against a root path that
    /// silently stopped existing.
    ///
    /// # Panics
    /// If fewer than `min` files were found.
    pub fn assert_scanned_at_least(&self, min: usize) {
        let n = self.files().len();
        assert!(
            n >= min,
            "the production scan found only {n} .rs files (expected >= {min}) — the roots are \
             probably wrong, and an empty scan passes every absence test vacuously"
        );
    }
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    if dir.is_file() {
        if dir.extension().is_some_and(|e| e == "rs") {
            out.push(dir.to_path_buf());
        }
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.filter_map(Result::ok) {
        let p = e.path();
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spline_src() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hymeko_spline/src")
    }

    #[test]
    fn the_scan_finds_the_spline_sources() {
        let s = ProductionScan::new([spline_src()]);
        // Three since 2026-08-29: `lib.rs`, `catmull_rom.rs`, `kochanek_bartels.rs`. It was four
        // until `clamp_contract.rs` was deleted with the selector. The guard fired on that change,
        // which is what it is for — an absence test over a shrinking file set passes vacuously.
        s.assert_scanned_at_least(3);
        assert!(s.files().iter().any(|p| p.ends_with("catmull_rom.rs")));
    }

    #[test]
    fn a_token_that_is_present_is_reported_with_its_line() {
        let s = ProductionScan::new([spline_src()]);
        let hits = s.hits("pub fn catmull_rom_forward");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].line_no > 1);
    }

    #[test]
    fn a_token_that_is_absent_passes() {
        ProductionScan::new([spline_src()]).assert_absent("ThisTokenDoesNotExistAnywhere");
    }

    #[test]
    #[should_panic(expected = "must not appear in production code")]
    fn a_token_that_is_present_fails() {
        ProductionScan::new([spline_src()]).assert_absent("pub fn catmull_rom_forward");
    }

    #[test]
    fn a_nonexistent_root_is_skipped_rather_than_panicking() {
        let s = ProductionScan::new([PathBuf::from("/nonexistent/path/xyz")]);
        assert!(s.files().is_empty());
        s.assert_absent("anything");
    }

    #[test]
    #[should_panic(expected = "passes every absence test vacuously")]
    fn an_empty_scan_is_caught_by_the_size_guard() {
        ProductionScan::new([PathBuf::from("/nonexistent/path/xyz")]).assert_scanned_at_least(1);
    }

    #[test]
    fn comment_lines_are_not_hits() {
        let dir = std::env::temp_dir().join("hymeko_numeric_vv_source_gate_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(
            dir.join("a.rs"),
            "// mentions Needle in prose\nlet x = 1;\n",
        )
        .expect("write");
        let s = ProductionScan::new([dir.clone()]);
        assert!(s.hits("Needle").is_empty());
        std::fs::write(dir.join("a.rs"), "let y = Needle;\n").expect("write");
        assert_eq!(s.hits("Needle").len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
