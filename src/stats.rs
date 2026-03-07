/// Persistent cumulative savings counter.
///
/// Stored at `~/.imptokens/stats.json`.  All operations are best-effort —
/// a failure to read or write never propagates to the caller.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CumulativeStats {
    pub total_runs: u64,
    pub total_original_tokens: u64,
    pub total_kept_tokens: u64,
}

impl CumulativeStats {
    pub fn total_saved(&self) -> u64 {
        self.total_original_tokens.saturating_sub(self.total_kept_tokens)
    }

    pub fn avg_reduction_pct(&self) -> f64 {
        if self.total_original_tokens == 0 {
            return 0.0;
        }
        (1.0 - self.total_kept_tokens as f64 / self.total_original_tokens as f64) * 100.0
    }
}

fn stats_path() -> Option<PathBuf> {
    #[cfg(windows)]
    let home = std::env::var("USERPROFILE").ok()?;
    #[cfg(not(windows))]
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".imptokens").join("stats.json"))
}

pub fn load() -> CumulativeStats {
    let path = match stats_path() {
        Some(p) => p,
        None => return CumulativeStats::default(),
    };
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return CumulativeStats::default(),
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

/// Record a single compression run.  Silently ignores all I/O errors.
pub fn record(original: u64, kept: u64) {
    let path = match stats_path() {
        Some(p) => p,
        None => return,
    };
    let mut stats = load();
    stats.total_runs += 1;
    stats.total_original_tokens += original;
    stats.total_kept_tokens += kept;
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&stats).unwrap_or_default());
}
