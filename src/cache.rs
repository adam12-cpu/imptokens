/// Simulated provider-side prompt caching.
///
/// A cache "hit" occurs when the prompt's prefix (up to `cache_window`
/// estimated tokens) matches a prefix seen in a previous call within the same
/// simulation run.  This mirrors the deterministic prefix-hashing used in
/// token-compression.mjs.
use std::collections::HashMap;

use crate::sentence::{estimate_tokens, fnv1a_hash, tokenize};

/// Result for a single prompt in a cache simulation run.
pub struct CacheEntry {
    /// Index of the prompt in the input slice.
    pub prompt_index: usize,
    /// Whether this prompt's prefix matched a previously-seen hash.
    pub is_hit: bool,
    /// Estimated total token count of the prompt.
    pub tokens: usize,
    /// Tokens billed for this prompt (0 on a cache hit, `tokens` on a miss).
    pub billable_tokens: usize,
    /// Hex string of the FNV-1a hash of the prefix.
    pub prefix_hash: String,
}

/// Simulate provider-side prompt caching for a sequence of prompts.
///
/// For each prompt the first `cache_window` estimated tokens are extracted
/// as a prefix, hashed deterministically, and compared against a running set
/// of seen hashes.  A hit means the prefix was already seen.
pub fn simulate_token_cache(prompts: &[&str], cache_window: usize) -> Vec<CacheEntry> {
    let mut seen: HashMap<String, bool> = HashMap::new();
    let mut results = Vec::with_capacity(prompts.len());

    for (i, prompt) in prompts.iter().enumerate() {
        let tokens = estimate_tokens(prompt);
        let prefix = extract_token_prefix(prompt, cache_window);
        let hash = format!("{:016x}", fnv1a_hash(&prefix));

        let is_hit = seen.contains_key(&hash);
        seen.insert(hash.clone(), true);

        results.push(CacheEntry {
            prompt_index: i,
            is_hit,
            tokens,
            billable_tokens: if is_hit { 0 } else { tokens },
            prefix_hash: hash,
        });
    }

    results
}

/// Extract the first `token_count` tokens from `text` as a string slice.
///
/// Returns an owned `String` of the text up to (and including) the last
/// character of the `token_count`-th token.
fn extract_token_prefix(text: &str, token_count: usize) -> String {
    if text.is_empty() || token_count == 0 {
        return String::new();
    }
    let tokens = tokenize(text);
    let n = token_count.min(tokens.len());
    if n == 0 {
        return String::new();
    }
    // Each &str in `tokens` is a slice of `text`, so we can recover the byte
    // offset of the end of the last kept token.
    let last = tokens[n - 1];
    let end = (last.as_ptr() as usize - text.as_ptr() as usize) + last.len();
    text[..end].to_string()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_prompt_is_always_miss() {
        let results = simulate_token_cache(&["hello world"], 600);
        assert!(!results[0].is_hit);
        assert_eq!(results[0].billable_tokens, results[0].tokens);
    }

    #[test]
    fn identical_prefix_is_hit() {
        let shared = "This is a shared system prompt. ";
        let p1 = format!("{}context one", shared);
        let p2 = format!("{}context two", shared);
        // With a small window the prefix (shared part) is the same.
        let results =
            simulate_token_cache(&[p1.as_str(), p2.as_str()], 6);
        assert!(!results[0].is_hit);
        assert!(results[1].is_hit, "second prompt with same prefix should be a hit");
        assert_eq!(results[1].billable_tokens, 0);
    }

    #[test]
    fn different_prefixes_are_misses() {
        let results = simulate_token_cache(&["foo bar", "baz qux"], 600);
        assert!(!results[0].is_hit);
        assert!(!results[1].is_hit);
    }

    #[test]
    fn window_limit_respected() {
        // Two prompts that differ only beyond the window should both hash to
        // the same prefix and the second should be a hit.
        let base = "word ".repeat(5); // 5 tokens
        let p1 = format!("{}UNIQUE_A", base);
        let p2 = format!("{}UNIQUE_B", base);
        let results = simulate_token_cache(&[p1.as_str(), p2.as_str()], 5);
        assert!(results[1].is_hit);
    }

    #[test]
    fn prefix_only_matching_not_suffix() {
        // If prompts share a suffix but not a prefix they must NOT hit.
        let results = simulate_token_cache(&["alpha beta gamma", "delta epsilon gamma"], 3);
        assert!(!results[1].is_hit);
    }
}
