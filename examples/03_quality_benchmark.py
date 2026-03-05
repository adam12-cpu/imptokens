#!/usr/bin/env python3
"""
Example 03: Quality benchmark
Measures how well compression preserves information across multiple text types
and compression ratios/reductions. Reports:
  - Token reduction
  - Key-phrase survival (did the main concepts make it through?)
  - Compression latency (ms)
  - Per-category best ratio recommendation

Supports two compression modes:
  logprob  — logprob-based token filtering (requires model, default thresholds)
  sentence — query-relevant sentence extraction (no model, zero-dependency)

Usage:
    python3 examples/03_quality_benchmark.py
    python3 examples/03_quality_benchmark.py --mode sentence
    python3 examples/03_quality_benchmark.py --mode sentence --reductions 0.05 0.1 0.2
    python3 examples/03_quality_benchmark.py --mode logprob --ratios 0.8 0.6 0.4
    python3 examples/03_quality_benchmark.py --cases 4   # quick run, first 4 only
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

_DEFAULT_BIN = shutil.which("imptokens") or "./target/release/imptokens"

# ── Test cases ───────────────────────────────────────────────────────────────
# Each case has a `category` tag used to group the summary recommendation.
CASES = [
    # ── Technical prose ─────────────────────────────────────────────────────
    {
        "label": "Transformer architecture overview",
        "category": "technical-prose",
        "text": (
            "Transformers use self-attention to compute weighted sums of value vectors, "
            "where weights are derived from queries and keys via scaled dot-product attention. "
            "The multi-head variant projects inputs into multiple subspaces, enabling the model "
            "to attend to information from different representation subspaces simultaneously. "
            "Residual connections and layer normalization stabilize training of deep networks. "
            "Positional encodings inject sequence-order information because the self-attention "
            "operation is permutation-invariant by construction. The feed-forward sublayer "
            "consists of two linear transformations with a ReLU activation in between, applied "
            "identically and independently to each position. During pre-training the model "
            "minimizes cross-entropy loss over masked tokens; during fine-tuning a task-specific "
            "head is appended and the whole stack is updated end-to-end with a smaller learning rate."
        ),
        "key_phrases": ["self-attention", "queries", "keys", "multi-head", "residual",
                        "positional", "feed-forward", "cross-entropy"],
    },
    {
        "label": "Kubernetes pod scheduling explained",
        "category": "technical-prose",
        "text": (
            "When a pod is created in Kubernetes the scheduler selects a node by filtering "
            "all available nodes through predicates such as resource availability, node affinity, "
            "and taints and tolerations. Nodes that pass filtering are then ranked by priority "
            "functions including balanced resource usage and least-allocated heuristics. The "
            "scheduler binds the pod to the highest-scoring node by writing a Binding object to "
            "the API server. The kubelet on that node watches for pods assigned to it and starts "
            "the container runtime to pull images and launch containers. If the scheduler cannot "
            "find a feasible node the pod remains in Pending state and the cluster autoscaler "
            "may provision additional nodes to satisfy the request. Pod disruption budgets prevent "
            "voluntary disruptions from evicting too many replicas at once, ensuring availability "
            "during rolling updates and node drains."
        ),
        "key_phrases": ["scheduler", "predicates", "node affinity", "taints", "kubelet",
                        "Pending", "autoscaler", "disruption budget"],
    },
    # ── Repetitive / boilerplate documentation ───────────────────────────────
    {
        "label": "SDK function reference (repetitive)",
        "category": "repetitive-docs",
        "text": (
            "To install the package, run pip install mypackage. "
            "After installation, import the package with import mypackage. "
            "The package provides the following functions. "
            "mypackage.compress(data, level=6) — compresses data using zlib at the given level. "
            "mypackage.decompress(data) — decompresses data previously compressed by compress(). "
            "mypackage.validate(data) — validates that data is well-formed before processing. "
            "mypackage.encode(data, encoding='utf-8') — encodes a string to bytes. "
            "mypackage.decode(data, encoding='utf-8') — decodes bytes back to a string. "
            "mypackage.checksum(data) — computes a CRC32 checksum over the input bytes. "
            "All functions return None on invalid input and raise ValueError for type errors. "
            "All functions are thread-safe and can be called concurrently without external locking. "
            "All functions accept both bytes and bytearray as input."
        ),
        "key_phrases": ["compress", "decompress", "validate", "encode", "decode", "checksum",
                        "ValueError", "thread-safe"],
    },
    {
        "label": "CLI help text (highly repetitive)",
        "category": "repetitive-docs",
        "text": (
            "usage: deploy.sh [-h] [--env ENV] [--region REGION] [--dry-run] [--force] service\n"
            "\n"
            "Deploy a service to the target environment.\n"
            "\n"
            "positional arguments:\n"
            "  service               Name of the service to deploy. Must match a directory under services/.\n"
            "\n"
            "optional arguments:\n"
            "  -h, --help            Show this help message and exit.\n"
            "  --env ENV             Target environment: dev, staging, or prod. Default: dev.\n"
            "  --region REGION       AWS region to deploy to. Default: us-east-1.\n"
            "  --dry-run             Print the deployment plan without applying changes.\n"
            "  --force               Skip confirmation prompts. Use with caution in prod.\n"
            "  --timeout TIMEOUT     Deployment timeout in seconds. Default: 300.\n"
            "  --replicas REPLICAS   Number of replicas to run. Default: derived from env config.\n"
            "  --image-tag TAG       Docker image tag to deploy. Default: latest.\n"
            "  --rollback            Roll back to the previous deployment on failure.\n"
            "  --notify CHANNEL      Slack channel to notify on deployment completion.\n"
            "\n"
            "examples:\n"
            "  deploy.sh api-server --env staging --dry-run\n"
            "  deploy.sh worker --env prod --force --image-tag v1.4.2\n"
            "  deploy.sh frontend --env dev --notify '#deployments'\n"
        ),
        "key_phrases": ["--env", "--region", "--dry-run", "--force", "--rollback",
                        "staging", "prod", "replicas"],
    },
    # ── Git diffs ────────────────────────────────────────────────────────────
    {
        "label": "Git diff — small refactor",
        "category": "git-diff",
        "text": (
            "diff --git a/src/model.py b/src/model.py\n"
            "index 3f4a2b1..8c9d0e2 100644\n"
            "--- a/src/model.py\n"
            "+++ b/src/model.py\n"
            "@@ -42,7 +42,9 @@ class Transformer(nn.Module):\n"
            "     def forward(self, x):\n"
            "-        return self.layers(x)\n"
            "+        x = self.embed(x)\n"
            "+        x = self.layers(x)\n"
            "+        return self.head(x)\n"
            "@@ -58,4 +60,6 @@ class Transformer(nn.Module):\n"
            "     def reset_parameters(self):\n"
            "         for layer in self.layers:\n"
            "-            layer.reset()\n"
            "+            layer.reset_parameters()\n"
            "+        if self.head is not None:\n"
            "+            nn.init.xavier_uniform_(self.head.weight)\n"
        ),
        "key_phrases": ["forward", "embed", "layers", "head", "xavier_uniform_",
                        "reset_parameters"],
    },
    {
        "label": "Git diff — security fix",
        "category": "git-diff",
        "text": (
            "diff --git a/src/auth.py b/src/auth.py\n"
            "index a1b2c3d..e4f5a6b 100644\n"
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -12,8 +12,9 @@ import hashlib\n"
            " import hmac\n"
            "+import secrets\n"
            "\n"
            "@@ -31,7 +32,7 @@ class SessionManager:\n"
            "     def create_session(self, user_id: int) -> str:\n"
            "-        token = hashlib.sha256(str(user_id + time.time()).encode()).hexdigest()\n"
            "+        token = secrets.token_hex(32)\n"
            "         self._store[token] = {'user_id': user_id, 'created': time.time()}\n"
            "         return token\n"
            "\n"
            "@@ -45,6 +46,9 @@ class SessionManager:\n"
            "     def validate_token(self, token: str, expected: str) -> bool:\n"
            "-        return token == expected\n"
            "+        return hmac.compare_digest(\n"
            "+            token.encode('utf-8'),\n"
            "+            expected.encode('utf-8'),\n"
            "+        )\n"
        ),
        "key_phrases": ["secrets", "token_hex", "hmac", "compare_digest",
                        "timing attack", "sha256"],
    },
    # ── Error logs / stack traces ─────────────────────────────────────────────
    {
        "label": "PyTorch training crash",
        "category": "error-log",
        "text": (
            "Traceback (most recent call last):\n"
            "  File 'train.py', line 142, in run_epoch\n"
            "    loss = criterion(outputs, targets)\n"
            "  File 'loss.py', line 67, in forward\n"
            "    return F.cross_entropy(input, target, reduction=self.reduction)\n"
            "RuntimeError: Expected input batch_size (32) to match target batch_size (16). "
            "Check that DataLoader drop_last=True or that batch sizes are consistent.\n"
            "\n"
            "The above exception was the direct cause of the following exception:\n"
            "\n"
            "Traceback (most recent call last):\n"
            "  File 'runner.py', line 89, in main\n"
            "    trainer.fit(model, datamodule)\n"
            "  File 'trainer.py', line 145, in fit\n"
            "    self._run_epoch(model, loader, optimizer, phase='train')\n"
            "  File 'trainer.py', line 178, in _run_epoch\n"
            "    batch_loss = self._forward_step(batch)\n"
            "RuntimeError: Expected input batch_size (32) to match target batch_size (16).\n"
            "\n"
            "System: torch==2.1.0, cuda==12.1, GPU=A100-80GB\n"
            "Config:  batch_size=32, drop_last=False, num_workers=4\n"
            "Last good commit: a3f2c89 'halve batch size for memory'\n"
        ),
        "key_phrases": ["RuntimeError", "batch_size", "32", "16", "drop_last",
                        "cross_entropy", "trainer"],
    },
    {
        "label": "Distributed system error log",
        "category": "error-log",
        "text": (
            "2025-03-04 03:14:23 INFO  [main] Database connection pool initialized (min=5, max=20)\n"
            "2025-03-04 03:14:23 INFO  [main] Redis connection established at redis:6379\n"
            "2025-03-04 03:14:45 INFO  [worker-1] Processing batch job: daily_cleanup\n"
            "2025-03-04 03:14:45 INFO  [worker-1] Found 12,847 expired sessions to delete\n"
            "2025-03-04 03:15:01 WARN  [worker-1] Slow query (2340ms): DELETE FROM sessions WHERE expires_at < NOW()\n"
            "2025-03-04 03:15:01 WARN  [worker-1] Slow query (2340ms): DELETE FROM sessions WHERE expires_at < NOW()\n"
            "2025-03-04 03:15:01 WARN  [worker-1] Slow query (2340ms): DELETE FROM sessions WHERE expires_at < NOW()\n"
            "2025-03-04 03:15:04 ERROR [worker-1] Database connection timeout after 30000ms\n"
            "2025-03-04 03:15:04 ERROR [worker-1] Retrying (1/3)...\n"
            "2025-03-04 03:15:34 ERROR [worker-1] Database connection timeout after 30000ms\n"
            "2025-03-04 03:15:34 ERROR [worker-1] Retrying (2/3)...\n"
            "2025-03-04 03:15:34 WARN  [http-4] Request queue depth: 847 (threshold: 500)\n"
            "2025-03-04 03:15:35 WARN  [http-5] Request queue depth: 1203 (threshold: 500)\n"
            "2025-03-04 03:16:04 ERROR [worker-1] Database connection timeout after 30000ms\n"
            "2025-03-04 03:16:04 ERROR [worker-1] Max retries reached. Aborting job: daily_cleanup\n"
            "2025-03-04 03:16:04 FATAL [main] Connection pool exhausted (0/20 available)\n"
            "2025-03-04 03:16:04 FATAL [main] Initiating graceful shutdown\n"
        ),
        "key_phrases": ["connection pool", "exhausted", "Slow query", "sessions",
                        "timeout", "FATAL", "30000ms"],
    },
    # ── Source code ───────────────────────────────────────────────────────────
    {
        "label": "Python code with docstrings",
        "category": "source-code",
        "text": (
            'class RateLimiter:\n'
            '    """Token-bucket rate limiter for API clients.\n'
            '\n'
            '    Allows `capacity` requests to burst immediately, then replenishes\n'
            '    at `rate` requests per second. Thread-safe via a reentrant lock.\n'
            '\n'
            '    Args:\n'
            '        rate: Replenishment rate in requests per second.\n'
            '        capacity: Maximum burst size (token bucket capacity).\n'
            '    """\n'
            '\n'
            '    def __init__(self, rate: float, capacity: float):\n'
            '        self.rate = rate\n'
            '        self.capacity = capacity\n'
            '        self._tokens = capacity\n'
            '        self._last = time.monotonic()\n'
            '        self._lock = threading.RLock()\n'
            '\n'
            '    def acquire(self, n: float = 1.0) -> float:\n'
            '        """Block until `n` tokens are available; return wait time in seconds.\n'
            '\n'
            '        Uses a spin-sleep loop with exponential back-off to avoid busy-waiting.\n'
            '        Raises ValueError if n exceeds capacity (would never be satisfiable).\n'
            '        """\n'
            '        if n > self.capacity:\n'
            '            raise ValueError(f"n={n} exceeds capacity={self.capacity}")\n'
            '        waited = 0.0\n'
            '        while True:\n'
            '            with self._lock:\n'
            '                now = time.monotonic()\n'
            '                elapsed = now - self._last\n'
            '                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)\n'
            '                self._last = now\n'
            '                if self._tokens >= n:\n'
            '                    self._tokens -= n\n'
            '                    return waited\n'
            '            sleep_for = (n - self._tokens) / self.rate\n'
            '            time.sleep(sleep_for)\n'
            '            waited += sleep_for\n'
        ),
        "key_phrases": ["RateLimiter", "token-bucket", "acquire", "capacity",
                        "RLock", "monotonic", "ValueError"],
    },
    {
        "label": "Rust async code snippet",
        "category": "source-code",
        "text": (
            "use std::time::Duration;\n"
            "use tokio::sync::Semaphore;\n"
            "use tokio::time::timeout;\n"
            "\n"
            "/// Connection pool for async database access.\n"
            "pub struct Pool {\n"
            "    semaphore: Arc<Semaphore>,\n"
            "    connections: Arc<Mutex<Vec<Connection>>>,\n"
            "    connect_timeout: Duration,\n"
            "}\n"
            "\n"
            "impl Pool {\n"
            "    pub async fn acquire(&self) -> Result<PooledConn, PoolError> {\n"
            "        let permit = timeout(self.connect_timeout, self.semaphore.acquire())\n"
            "            .await\n"
            "            .map_err(|_| PoolError::Timeout)?\n"
            "            .map_err(|_| PoolError::Closed)?;\n"
            "        let conn = self.connections.lock().await.pop()\n"
            "            .ok_or(PoolError::Exhausted)?;\n"
            "        Ok(PooledConn { conn, permit, pool: Arc::clone(&self.connections) })\n"
            "    }\n"
            "\n"
            "    pub async fn release(&self, conn: Connection) {\n"
            "        self.connections.lock().await.push(conn);\n"
            "    }\n"
            "}\n"
        ),
        "key_phrases": ["Semaphore", "acquire", "timeout", "PoolError",
                        "Exhausted", "Arc", "Mutex"],
    },
    # ── Structured / JSON data ────────────────────────────────────────────────
    {
        "label": "Kubernetes deployment manifest",
        "category": "structured-data",
        "text": (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: api-server\n"
            "  namespace: production\n"
            "  labels:\n"
            "    app: api-server\n"
            "    version: v2.4.1\n"
            "spec:\n"
            "  replicas: 6\n"
            "  selector:\n"
            "    matchLabels:\n"
            "      app: api-server\n"
            "  strategy:\n"
            "    type: RollingUpdate\n"
            "    rollingUpdate:\n"
            "      maxSurge: 2\n"
            "      maxUnavailable: 1\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "      - name: api-server\n"
            "        image: registry.example.com/api-server:v2.4.1\n"
            "        ports:\n"
            "        - containerPort: 8080\n"
            "        resources:\n"
            "          requests:\n"
            "            cpu: '500m'\n"
            "            memory: '512Mi'\n"
            "          limits:\n"
            "            cpu: '2000m'\n"
            "            memory: '2Gi'\n"
            "        livenessProbe:\n"
            "          httpGet:\n"
            "            path: /healthz\n"
            "            port: 8080\n"
            "          initialDelaySeconds: 15\n"
            "          periodSeconds: 10\n"
            "        readinessProbe:\n"
            "          httpGet:\n"
            "            path: /ready\n"
            "            port: 8080\n"
            "          initialDelaySeconds: 5\n"
            "          periodSeconds: 5\n"
            "      terminationGracePeriodSeconds: 60\n"
        ),
        "key_phrases": ["replicas", "RollingUpdate", "maxSurge", "livenessProbe",
                        "readinessProbe", "512Mi", "2Gi", "production"],
    },
    {
        "label": "OpenAPI / JSON schema (repetitive structure)",
        "category": "structured-data",
        "text": (
            '{\n'
            '  "openapi": "3.0.3",\n'
            '  "info": { "title": "Widget API", "version": "2.0.0" },\n'
            '  "paths": {\n'
            '    "/widgets": {\n'
            '      "get": {\n'
            '        "summary": "List widgets",\n'
            '        "parameters": [\n'
            '          { "name": "page", "in": "query", "schema": { "type": "integer", "default": 1 } },\n'
            '          { "name": "per_page", "in": "query", "schema": { "type": "integer", "default": 20, "maximum": 100 } },\n'
            '          { "name": "sort", "in": "query", "schema": { "type": "string", "enum": ["created_at", "name", "updated_at"] } }\n'
            '        ],\n'
            '        "responses": {\n'
            '          "200": { "description": "Success", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/WidgetList" } } } },\n'
            '          "401": { "description": "Unauthorized" },\n'
            '          "429": { "description": "Rate limit exceeded" }\n'
            '        }\n'
            '      },\n'
            '      "post": {\n'
            '        "summary": "Create widget",\n'
            '        "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/WidgetCreate" } } } },\n'
            '        "responses": {\n'
            '          "201": { "description": "Created" },\n'
            '          "400": { "description": "Validation error" },\n'
            '          "401": { "description": "Unauthorized" }\n'
            '        }\n'
            '      }\n'
            '    }\n'
            '  }\n'
            '}\n'
        ),
        "key_phrases": ["openapi", "per_page", "429", "WidgetList", "requestBody",
                        "created_at", "maximum"],
    },
    # ── Long narrative prose ──────────────────────────────────────────────────
    {
        "label": "Wikipedia-style technical article",
        "category": "narrative-prose",
        "text": (
            "The Apollo 11 mission, launched on July 16, 1969, was the first crewed lunar landing "
            "in history. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the "
            "Apollo Lunar Module Eagle on the Moon on July 20, 1969, at 20:17 UTC, while Command "
            "Module Pilot Michael Collins orbited above in the Command Module Columbia. Armstrong "
            "became the first person to step onto the lunar surface six hours and 39 minutes later "
            "on July 21 at 02:56 UTC; Aldrin joined him 19 minutes later. They spent about two "
            "and a quarter hours together outside the spacecraft, and collected 47.5 pounds "
            "(21.5 kg) of lunar material to bring back to Earth. After 21 hours and 36 minutes "
            "on the lunar surface the module ascended and rejoined Columbia in orbit. "
            "The mission fulfilled a national goal proposed in 1961 by President John F. Kennedy: "
            "performing a crewed lunar landing and returning safely to Earth before the end of "
            "the decade. The crew landed in the Pacific Ocean on July 24 and were recovered by "
            "USS Hornet. The three astronauts were placed in quarantine for 21 days as a "
            "precaution against possible contamination from lunar material. All three men received "
            "the Presidential Medal of Freedom from President Richard Nixon."
        ),
        "key_phrases": ["Apollo 11", "Armstrong", "Aldrin", "July 20", "Eagle",
                        "21.5 kg", "Kennedy", "Pacific Ocean", "quarantine"],
    },
    {
        "label": "Meeting notes (mixed structure)",
        "category": "narrative-prose",
        "text": (
            "Q1 2025 Architecture Review — Meeting Notes\n"
            "Date: 2025-03-05  |  Attendees: Alice (EM), Bob (backend), Carol (infra), Dave (security)\n"
            "\n"
            "## Agenda items\n"
            "\n"
            "1. Database migration status\n"
            "   - Bob: Postgres 14 → 16 upgrade is 80% done. Blocking issue: pgvector extension "
            "     must be rebuilt for 16. ETA: end of sprint.\n"
            "   - Carol: Downtime window proposed for March 15, 02:00-04:00 UTC. Need approval.\n"
            "   - ACTION: Bob to file change request by EOD Friday.\n"
            "\n"
            "2. Auth service refactor\n"
            "   - Dave: JWT expiry currently set to 7 days — flagged as too long. Recommend 1h "
            "     access token + 30d refresh token pattern.\n"
            "   - Alice: Approved in principle. Bob to implement, targeting Q1 end.\n"
            "   - ACTION: Dave to update threat model document.\n"
            "\n"
            "3. Observability gap\n"
            "   - Carol: Worker queues have no SLO defined. P99 latency spikes to 12s undetected.\n"
            "   - ACTION: Carol to add queue-depth and latency dashboards in Grafana by next sprint.\n"
            "\n"
            "Next meeting: March 19, same time.\n"
        ),
        "key_phrases": ["Postgres", "pgvector", "March 15", "JWT", "refresh token",
                        "P99", "Grafana", "ACTION"],
    },
]

DEFAULT_RATIOS = [0.7, 0.5, 0.3]
DEFAULT_REDUCTIONS = [0.05, 0.1, 0.2, 0.45]  # sentence mode: fraction to remove


@dataclass
class Result:
    label: str
    category: str
    ratio_target: float      # keep-ratio (logprob) or target-reduction (sentence)
    n_orig: int
    n_kept: int
    compression_ratio: float
    key_phrase_survival: float
    latency_ms: float
    compressed_text: str


def run_logprob(text: str, ratio: float, binary: str) -> tuple[dict, float]:
    """Run logprob compression; returns (data, latency_ms)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [binary, "--keep-ratio", str(ratio), "--output-format", "json"],
        input=text, capture_output=True, text=True,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout), latency_ms


def run_sentence(text: str, reduction: float, query: str, binary: str) -> tuple[dict, float]:
    """Run sentence-mode compression; returns (data, latency_ms)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [binary, "--sentence-mode",
         f"--target-reduction={reduction}",
         f"--query={query}",
         "--output-format", "json"],
        input=text, capture_output=True, text=True,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:200]}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout), latency_ms


def key_phrase_survival(phrases: list, compressed: str) -> float:
    found = sum(1 for p in phrases if p.lower() in compressed.lower())
    return found / len(phrases) if phrases else 1.0


def benchmark(cases: list, ratios: list, mode: str, binary: str) -> list:
    results = []
    total = len(cases) * len(ratios)
    i = 0
    for case in cases:
        for ratio in ratios:
            i += 1
            param_label = f"reduction={ratio}" if mode == "sentence" else f"ratio={ratio}"
            print(f"  [{i:>2}/{total}] {case['label']:<45} {param_label}…",
                  end=" ", flush=True)
            if mode == "sentence":
                # Use key_phrases as the query for sentence scoring
                query = " ".join(case["key_phrases"])
                data, latency_ms = run_sentence(case["text"], ratio, query, binary)
            else:
                data, latency_ms = run_logprob(case["text"], ratio, binary)
            survival = key_phrase_survival(case["key_phrases"], data["compressed_text"])
            results.append(Result(
                label=case["label"],
                category=case["category"],
                ratio_target=ratio,
                n_orig=data["n_original"],
                n_kept=data["n_kept"],
                compression_ratio=data["compression_ratio"],
                key_phrase_survival=survival,
                latency_ms=latency_ms,
                compressed_text=data["compressed_text"],
            ))
            print(f"✓  {data['n_kept']:>3}/{data['n_original']:<3} tokens  "
                  f"phrases={survival*100:.0f}%  {latency_ms:.0f}ms")
    return results


def report(results: list, ratios: list, mode: str):
    W = 80
    param_name = "Reduction" if mode == "sentence" else "Ratio"
    print("\n" + "=" * W)
    print(f"  imptokens — Quality Benchmark Results  [{mode} mode]")
    print("=" * W)

    # Group by label
    by_label: dict = {}
    for r in results:
        by_label.setdefault(r.label, []).append(r)

    for label, rs in by_label.items():
        cat = rs[0].category
        print(f"\n  [{cat}]  {label}")
        print(f"  {'─' * 68}")
        print(f"  {param_name:>10}  {'Actual':>7}  {'Tokens':>11}  {'Latency':>8}  Phrases")
        for r in sorted(rs, key=lambda x: x.ratio_target):
            actual_pct = r.compression_ratio * 100
            bar = "█" * int(r.key_phrase_survival * 10) + "░" * (10 - int(r.key_phrase_survival * 10))
            print(f"  {r.ratio_target*100:>9.0f}%  {actual_pct:>6.1f}%  "
                  f"{r.n_kept:>4}/{r.n_orig:<4}  "
                  f"{r.latency_ms:>6.0f}ms  "
                  f"{bar} {r.key_phrase_survival*100:.0f}%")

    # Category summary
    print("\n" + "─" * W)
    print(f"  Category summary  ({param_name.lower()} = best savings×survival tradeoff)")
    print(f"  {'─' * 68}")
    print(f"  {'Category':<22}  {param_name:>9}  {'Avg savings':>11}  {'Phrases':>8}  "
          f"{'Avg latency':>11}  Recommendation")

    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    for cat, rs in sorted(by_cat.items()):
        by_ratio: dict = {}
        for r in rs:
            by_ratio.setdefault(r.ratio_target, []).append(r)
        best_ratio = max(
            by_ratio.keys(),
            key=lambda rt: (
                (1 - sum(r.compression_ratio for r in by_ratio[rt]) / len(by_ratio[rt]))
                * (sum(r.key_phrase_survival for r in by_ratio[rt]) / len(by_ratio[rt]))
            )
        )
        avg_savings = 1 - sum(r.compression_ratio for r in by_ratio[best_ratio]) / len(by_ratio[best_ratio])
        avg_survival = sum(r.key_phrase_survival for r in by_ratio[best_ratio]) / len(by_ratio[best_ratio])
        avg_latency = sum(r.latency_ms for r in by_ratio[best_ratio]) / len(by_ratio[best_ratio])
        if avg_survival >= 0.90:
            rec = "safe at this setting"
        elif avg_survival >= 0.70:
            rec = "acceptable tradeoff"
        else:
            rec = "use smaller reduction" if mode == "sentence" else "use higher ratio"
        print(f"  {cat:<22}  {best_ratio*100:>8.0f}%  {avg_savings*100:>10.1f}%  "
              f"{avg_survival*100:>7.0f}%  {avg_latency:>9.0f}ms  {rec}")

    # Overall summary
    print("\n" + "─" * W)
    avg_reduction = 1 - sum(r.compression_ratio for r in results) / len(results)
    avg_survival = sum(r.key_phrase_survival for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    p50_latency = sorted(r.latency_ms for r in results)[len(results) // 2]
    p95_latency = sorted(r.latency_ms for r in results)[int(len(results) * 0.95)]
    print(f"  Cases: {len(by_label)} texts × {len(ratios)} params = {len(results)} runs  [{mode} mode]")
    print(f"  Average token reduction (all):        {avg_reduction*100:.1f}%")
    print(f"  Average key-phrase survival (all):    {avg_survival*100:.1f}%")
    print(f"  Compression latency — avg: {avg_latency:.0f}ms  p50: {p50_latency:.0f}ms  p95: {p95_latency:.0f}ms")
    print()
    print("  Tip: key phrases = domain-critical terms. High survival + high reduction = good.")
    print("=" * W)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="imptokens quality benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sentence mode derives the compression query from each case's key_phrases,\n"
            "so no external model or API key is required.\n\n"
            "Examples:\n"
            "  python3 examples/03_quality_benchmark.py --mode sentence\n"
            "  python3 examples/03_quality_benchmark.py --mode sentence --reductions 0.05 0.1 0.2\n"
            "  python3 examples/03_quality_benchmark.py --mode logprob --ratios 0.7 0.5 0.3\n"
        ),
    )
    ap.add_argument("--mode", choices=["sentence", "logprob"], default="sentence",
                    help="Compression mode (default: sentence)")
    ap.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                    metavar="R", help="keep-ratios for logprob mode (default: 0.7 0.5 0.3)")
    ap.add_argument("--reductions", type=float, nargs="+", default=DEFAULT_REDUCTIONS,
                    metavar="R",
                    help="target-reduction values for sentence mode (default: 0.05 0.1 0.2 0.45)")
    ap.add_argument("--cases", type=int, default=None,
                    help="Run only first N cases (for quick testing)")
    ap.add_argument("--binary", default=_DEFAULT_BIN, metavar="PATH",
                    help="Path to imptokens binary (default: auto-detect)")
    args = ap.parse_args()

    cases = CASES[:args.cases] if args.cases else CASES
    params = args.reductions if args.mode == "sentence" else args.ratios

    print("imptokens — Quality Benchmark")
    print(f"Binary: {args.binary}")
    print(f"Mode: {args.mode}")
    print(f"Cases: {len(cases)}, Params: {params}\n")
    print("Running compressions…")
    results = benchmark(cases, params, args.mode, args.binary)
    report(results, params, args.mode)
