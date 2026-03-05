/// Benchmark runner for sentence-level context compression.
///
/// Mirrors the `runBenchmark` function from token-compression.mjs:
/// - Accepts a `Workload` (JSON-compatible) of context/query request pairs.
/// - Memoizes `preprocess_context` when multiple requests share the same context.
/// - Simulates token caching for both raw and compressed prompt sequences.
/// - Returns aggregate statistics and per-request details.
use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::cache::simulate_token_cache;
use crate::sentence::{
    build_prompt, compress_preprocessed, estimate_tokens, fnv1a_hash, preprocess_context,
    CompressContextOptions, PreprocessedContext,
};

// ─── Workload types ──────────────────────────────────────────────────────────

/// A single compression request inside a workload.
#[derive(Deserialize)]
pub struct WorkloadRequest {
    pub id: String,
    pub context: String,
    pub query: String,
    #[serde(default)]
    pub expected_points: Vec<String>,
}

/// A benchmark workload: an optional system prompt plus a list of requests.
#[derive(Deserialize)]
pub struct Workload {
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub requests: Vec<WorkloadRequest>,
}

// ─── Result types ────────────────────────────────────────────────────────────

/// Per-request result.
#[derive(Serialize)]
pub struct RequestResult {
    pub id: String,
    /// Estimated tokens in the raw (uncompressed) prompt.
    pub raw_tokens: usize,
    /// Estimated tokens in the compressed prompt.
    pub compressed_tokens: usize,
    /// `compressed_tokens / raw_tokens` (0.0 if raw_tokens == 0).
    pub compression_ratio: f64,
    /// Whether the compressed prompt was a cache hit.
    pub cache_hit: bool,
    /// Billable tokens (0 on cache hit, `compressed_tokens` on miss).
    pub billable_tokens: usize,
    /// Wall-clock time spent on compression, in milliseconds.
    pub compression_latency_ms: f64,
}

/// Aggregate benchmark result.
#[derive(Serialize)]
pub struct BenchmarkResult {
    pub n_requests: usize,
    pub raw_input_tokens: usize,
    pub compressed_input_tokens: usize,
    /// `1 - compressed / raw` (0.0 if raw == 0).
    pub input_token_reduction: f64,
    pub raw_billable_tokens: usize,
    pub compressed_billable_tokens: usize,
    pub billable_reduction: f64,
    /// Fraction of *raw* prompts that would have been cache hits.
    pub raw_cache_hit_rate: f64,
    pub avg_compression_latency_ms: f64,
    pub requests: Vec<RequestResult>,
}

// ─── Options ─────────────────────────────────────────────────────────────────

pub struct BenchmarkOptions {
    pub compress: CompressContextOptions,
    /// Prefix-window size (in tokens) used by the cache simulator.
    pub cache_window: usize,
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        Self { compress: CompressContextOptions::default(), cache_window: 600 }
    }
}

// ─── Benchmark runner ────────────────────────────────────────────────────────

/// Run the benchmark over `workload` using `opts`.
///
/// Preprocessing (sentence splitting + dedup + feature extraction) is memoized
/// per unique context string so that workloads with repeated contexts pay the
/// preprocessing cost only once.
pub fn run_benchmark(workload: &Workload, opts: BenchmarkOptions) -> BenchmarkResult {
    let sys = workload.system_prompt.as_deref();

    // Memoize PreprocessedContext by context FNV-1a hash.
    let mut preprocess_cache: HashMap<u64, PreprocessedContext> = HashMap::new();

    let mut request_results: Vec<RequestResult> = Vec::with_capacity(workload.requests.len());
    let mut raw_prompts: Vec<String> = Vec::with_capacity(workload.requests.len());
    let mut compressed_prompts: Vec<String> = Vec::with_capacity(workload.requests.len());

    for req in &workload.requests {
        let ctx_hash = fnv1a_hash(&req.context);

        // Ensure this context is preprocessed (memoized).
        if !preprocess_cache.contains_key(&ctx_hash) {
            preprocess_cache.insert(ctx_hash, preprocess_context(&req.context));
        }
        let pre = preprocess_cache.get(&ctx_hash).unwrap();

        let start = Instant::now();
        let compressed = compress_preprocessed(pre, &req.query, &opts.compress);
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let raw_prompt = build_prompt(&req.context, &req.query, sys);
        let compressed_prompt = build_prompt(&compressed.compressed, &req.query, sys);

        let raw_tokens = estimate_tokens(&raw_prompt);
        let compressed_tokens = estimate_tokens(&compressed_prompt);

        raw_prompts.push(raw_prompt);
        compressed_prompts.push(compressed_prompt);

        request_results.push(RequestResult {
            id: req.id.clone(),
            raw_tokens,
            compressed_tokens,
            compression_ratio: if raw_tokens > 0 {
                compressed_tokens as f64 / raw_tokens as f64
            } else {
                0.0
            },
            cache_hit: false,       // filled after cache simulation
            billable_tokens: 0,     // filled after cache simulation
            compression_latency_ms: latency_ms,
        });
    }

    // Simulate cache for raw and compressed prompt sequences.
    let raw_refs: Vec<&str> = raw_prompts.iter().map(String::as_str).collect();
    let comp_refs: Vec<&str> = compressed_prompts.iter().map(String::as_str).collect();

    let raw_cache = simulate_token_cache(&raw_refs, opts.cache_window);
    let comp_cache = simulate_token_cache(&comp_refs, opts.cache_window);

    // Aggregate totals.
    let mut raw_input_tokens: usize = 0;
    let mut compressed_input_tokens: usize = 0;
    let mut raw_billable_tokens: usize = 0;
    let mut compressed_billable_tokens: usize = 0;
    let mut raw_cache_hits: usize = 0;
    let mut total_latency_ms: f64 = 0.0;

    for (i, rr) in request_results.iter_mut().enumerate() {
        rr.cache_hit = comp_cache[i].is_hit;
        rr.billable_tokens = comp_cache[i].billable_tokens;

        raw_input_tokens += raw_cache[i].tokens;
        compressed_input_tokens += comp_cache[i].tokens;
        raw_billable_tokens += raw_cache[i].billable_tokens;
        compressed_billable_tokens += comp_cache[i].billable_tokens;

        if raw_cache[i].is_hit {
            raw_cache_hits += 1;
        }
        total_latency_ms += rr.compression_latency_ms;
    }

    let n = workload.requests.len();

    BenchmarkResult {
        n_requests: n,
        raw_input_tokens,
        compressed_input_tokens,
        input_token_reduction: if raw_input_tokens > 0 {
            1.0 - compressed_input_tokens as f64 / raw_input_tokens as f64
        } else {
            0.0
        },
        raw_billable_tokens,
        compressed_billable_tokens,
        billable_reduction: if raw_billable_tokens > 0 {
            1.0 - compressed_billable_tokens as f64 / raw_billable_tokens as f64
        } else {
            0.0
        },
        raw_cache_hit_rate: if n > 0 { raw_cache_hits as f64 / n as f64 } else { 0.0 },
        avg_compression_latency_ms: if n > 0 { total_latency_ms / n as f64 } else { 0.0 },
        requests: request_results,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_workload(requests: Vec<(&str, &str, &str)>) -> Workload {
        Workload {
            system_prompt: None,
            requests: requests
                .into_iter()
                .map(|(id, ctx, q)| WorkloadRequest {
                    id: id.to_string(),
                    context: ctx.to_string(),
                    query: q.to_string(),
                    expected_points: vec![],
                })
                .collect(),
        }
    }

    #[test]
    fn empty_workload() {
        let w = make_workload(vec![]);
        let r = run_benchmark(&w, BenchmarkOptions::default());
        assert_eq!(r.n_requests, 0);
        assert_eq!(r.raw_input_tokens, 0);
        assert_eq!(r.avg_compression_latency_ms, 0.0);
    }

    #[test]
    fn token_math_consistency() {
        let ctx = "The capital of France is Paris. Paris is known for the Eiffel Tower. \
                   The tower was built in 1889. It is 330 metres tall. \
                   France is a country in Western Europe.";
        let w = make_workload(vec![("r1", ctx, "What is the capital of France?")]);
        let r = run_benchmark(&w, BenchmarkOptions::default());

        assert_eq!(r.n_requests, 1);
        // Compressed should be <= raw
        assert!(r.compressed_input_tokens <= r.raw_input_tokens);
        // Billable reduction is in [0, 1]
        assert!((0.0..=1.0).contains(&r.billable_reduction));
    }

    #[test]
    fn cache_hits_for_shared_context() {
        let ctx = "Alpha beta gamma delta epsilon zeta. \
                   Eta theta iota kappa lambda mu nu. \
                   Xi omicron pi rho sigma tau upsilon.";
        let w = make_workload(vec![
            ("r1", ctx, "alpha query"),
            ("r2", ctx, "beta query"),
            ("r3", ctx, "gamma query"),
        ]);
        let r = run_benchmark(&w, BenchmarkOptions { cache_window: 5, ..Default::default() });
        // First request is always a miss; subsequent ones with same prefix should hit.
        assert!(!r.requests[0].cache_hit);
        // At least one of the later requests should be a cache hit.
        let hits = r.requests.iter().filter(|rr| rr.cache_hit).count();
        assert!(hits >= 1, "repeated context should produce cache hits");
    }

    #[test]
    fn sum_verification() {
        let ctx_a = "The sun is a star. Stars produce energy via nuclear fusion.";
        let ctx_b = "Water covers 71 percent of Earth. Most water is in the oceans.";
        let w = make_workload(vec![
            ("r1", ctx_a, "sun star"),
            ("r2", ctx_b, "water earth"),
        ]);
        let r = run_benchmark(&w, BenchmarkOptions::default());

        let sum_raw: usize = r.requests.iter().map(|rr| rr.raw_tokens).sum();
        assert_eq!(sum_raw, r.raw_input_tokens);

        let sum_compressed: usize = r.requests.iter().map(|rr| rr.compressed_tokens).sum();
        assert_eq!(sum_compressed, r.compressed_input_tokens);
    }

    #[test]
    fn compression_cache_memoization() {
        // Same context used in two requests — preprocessing must run only once.
        // We can't directly observe the cache, but we verify both requests
        // compress correctly without panicking.
        let ctx = "Rust is a systems programming language. It emphasizes safety. \
                   Memory safety is achieved without a garbage collector.";
        let w = make_workload(vec![
            ("r1", ctx, "Rust language"),
            ("r2", ctx, "memory safety"),
        ]);
        let r = run_benchmark(&w, BenchmarkOptions::default());
        assert_eq!(r.n_requests, 2);
        for rr in &r.requests {
            assert!(rr.raw_tokens > 0);
        }
    }

    #[test]
    fn ratio_math() {
        let ctx = "The quick brown fox jumps over the lazy dog. \
                   Pack my box with five dozen liquor jugs.";
        let w = make_workload(vec![("r1", ctx, "fox dog")]);
        let r = run_benchmark(&w, BenchmarkOptions::default());
        let expected = 1.0 - r.compressed_input_tokens as f64 / r.raw_input_tokens as f64;
        let diff = (r.input_token_reduction - expected).abs();
        assert!(diff < 1e-9, "input_token_reduction does not match computed ratio");
    }

    #[test]
    fn raw_cache_hit_rate_range() {
        let w = make_workload(vec![("r1", "hello world", "hello")]);
        let r = run_benchmark(&w, BenchmarkOptions::default());
        assert!((0.0..=1.0).contains(&r.raw_cache_hit_rate));
    }
}
