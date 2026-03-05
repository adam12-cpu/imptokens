#!/usr/bin/env python3
"""
Example 06: imptokens × Claude — Answer Quality Benchmark

Proves that compressing context with imptokens does not meaningfully degrade
the quality of Claude's answers. Supports two compression modes:

  sentence  (default) — query-relevant sentence extraction. No model required.
                        Fast (~1-5 ms/context). Controlled by --target-reduction.
  logprob             — logprob-based token filtering. Requires llama.cpp model.
                        Controlled by --threshold.

For each test case the benchmark:
  1. Sends the FULL context to Claude and records the answer.
  2. Compresses the context and sends the compressed version.
  3. Asks Claude to score both answers against key facts.
  4. Prints a side-by-side report with latency.

Sentence mode (default):
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

    python3 examples/06_claude_quality_benchmark.py
    python3 examples/06_claude_quality_benchmark.py --target-reduction 0.1
    python3 examples/06_claude_quality_benchmark.py --model claude-sonnet-4-6
    python3 examples/06_claude_quality_benchmark.py --save-report quality_report.md

Sentence sweep (test multiple reductions, find the optimal cutoff):
    python3 examples/06_claude_quality_benchmark.py --target-reductions 0.05 0.1 0.2 0.45
    python3 examples/06_claude_quality_benchmark.py --target-reductions 0.05 0.1 0.2 --cases 2

Logprob mode (logprob threshold guide — more negative = more aggressive):
  -0.05  very light   ~10-20% tokens dropped
  -0.10  light        ~30-40% tokens dropped
  -0.20  moderate     ~40-50% tokens dropped
  -0.30  significant  ~45-55% tokens dropped
  -0.50  aggressive   ~50-65% tokens dropped

    python3 examples/06_claude_quality_benchmark.py --mode logprob --threshold -0.1
    python3 examples/06_claude_quality_benchmark.py --mode logprob --thresholds -0.05 -0.1 -0.2

Requirements:
    - ANTHROPIC_API_KEY environment variable
    - imptokens binary on PATH or at ./target/release/imptokens
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional, Literal

try:
    import anthropic
except ImportError:
    sys.exit("anthropic package not found. Run:  pip install anthropic")

# ── Test cases ────────────────────────────────────────────────────────────────
# Design principles for each case:
#   1. Long enough (800-1500 words) that compression makes a real difference
#   2. Clear filler vs. signal — the question targets one section of a larger doc
#   3. All key_facts are verbatim strings from the context (no inference needed)
#   4. Realistic: the kind of thing developers actually paste into Claude

CASES = [
    # ── Case 1: Long library README ─────────────────────────────────────────
    # Filler: installation, quickstart, model definitions, query builder, migrations,
    #         transactions, contributing section.
    # Signal: the "Connection Pooling Configuration" section at the end.
    {
        "label": "Library README — connection pool config",
        "question": (
            "What configuration options control connection pooling, "
            "and what are their defaults?"
        ),
        "context": textwrap.dedent("""\
            # dblink — Async Database Client for Python

            A high-performance async database client with automatic connection pooling,
            query building, and migration support. Built for production workloads.

            ## Why dblink?

            Most Python database clients were designed for synchronous workloads. As async
            Python has matured, the gap between sync and async database access has become a
            real performance bottleneck. dblink was built from the ground up to be async-first,
            using uvloop and asyncpg under the hood for PostgreSQL, and aiosqlite for SQLite.

            Key features:
            - Fully async/await interface
            - Automatic connection pooling with health checks
            - Fluent query builder (no raw SQL required, though raw SQL is supported)
            - Schema migrations with rollback support
            - Type-safe query results via dataclasses

            ## Installation

                pip install dblink

            For PostgreSQL support:

                pip install dblink[postgres]

            For MySQL support:

                pip install dblink[mysql]

            For SQLite (bundled by default, no extras needed):

                pip install dblink

            ## Quick Start

            Connect to a database and run your first query in five lines:

                import asyncio
                from dblink import Database

                async def main():
                    db = Database("postgresql://user:pass@localhost/mydb")
                    await db.connect()
                    rows = await db.fetch("SELECT id, name FROM users WHERE active = true")
                    for row in rows:
                        print(row.id, row.name)
                    await db.disconnect()

                asyncio.run(main())

            ## Defining Models

            dblink supports an optional model layer for structured table access:

                from dblink import Model, Column

                class User(Model):
                    __table__ = "users"
                    id: int = Column(primary_key=True)
                    name: str = Column()
                    email: str = Column(unique=True)
                    created_at: datetime = Column(default="now()")
                    active: bool = Column(default=True)

            Once defined, models support full CRUD operations:

                # Create
                user = await User.create(name="Alice", email="alice@example.com")

                # Read
                user = await User.get(id=1)
                users = await User.filter(active=True).all()

                # Update
                await user.update(name="Alicia")

                # Delete
                await user.delete()

            ## Query Builder

            For more complex queries, use the fluent query builder:

                results = await (
                    db.query("orders")
                    .select("id", "total", "status")
                    .where("status", "in", ["pending", "processing"])
                    .where("total", ">", 100)
                    .order_by("created_at", desc=True)
                    .limit(50)
                    .fetch()
                )

            You can also use raw SQL when needed:

                results = await db.fetch(
                    "SELECT u.name, COUNT(o.id) as order_count "
                    "FROM users u JOIN orders o ON u.id = o.user_id "
                    "WHERE o.created_at > $1 GROUP BY u.name",
                    cutoff_date,
                )

            ## Transactions

            Wrap multiple operations in a transaction using async context managers:

                async with db.transaction():
                    user = await User.create(name="Bob", email="bob@example.com")
                    await Order.create(user_id=user.id, total=49.99)
                    # Any exception here rolls back the entire transaction

            Nested transactions use SAVEPOINTs automatically. The outermost
            `async with db.transaction()` block owns the real transaction; inner
            blocks become SAVEPOINTs that can be rolled back independently.

            ## Migrations

            dblink includes a migration system. Create a migration:

                dblink migrate create "add users table"

            This generates a timestamped migration file in ./migrations/. Fill in
            the `up` and `down` coroutines:

                async def up(db):
                    await db.execute(
                        "CREATE TABLE users ("
                        "    id SERIAL PRIMARY KEY,"
                        "    name TEXT NOT NULL,"
                        "    email TEXT UNIQUE NOT NULL,"
                        "    created_at TIMESTAMPTZ DEFAULT now()"
                        ")"
                    )

                async def down(db):
                    await db.execute("DROP TABLE users")

            Apply and manage migrations:

                dblink migrate up        # apply all pending
                dblink migrate down      # roll back the last applied migration
                dblink migrate status    # show applied / pending list

            ## Connection Pooling Configuration

            By default, dblink manages a connection pool automatically. You can tune
            pool behaviour by passing keyword arguments to `Database()`:

                db = Database(
                    "postgresql://user:pass@localhost/mydb",
                    pool_size=10,
                    max_overflow=5,
                    pool_timeout=30,
                    pool_recycle=1800,
                    pool_pre_ping=True,
                )

            Option descriptions and defaults:

            pool_size (default: 5)
                Number of connections to keep open in the pool at all times. Increase
                this if your application sustains high concurrency.

            max_overflow (default: 10)
                Maximum number of extra connections allowed above pool_size during
                traffic spikes. These connections are closed as soon as they are
                returned to the pool. The total maximum open connections is
                pool_size + max_overflow.

            pool_timeout (default: 30)
                Seconds to wait for an available connection before raising
                PoolTimeout. Tune this to match your request timeout budget.

            pool_recycle (default: 3600)
                Seconds after which a connection is proactively replaced. Prevents
                errors from stale connections that were silently dropped by the
                database server or an intervening firewall. Set it slightly below
                your firewall's idle-connection timeout.

            pool_pre_ping (default: False)
                When True, dblink issues a lightweight SELECT 1 before handing a
                connection to your code. This adds a small round-trip overhead on
                every checkout but eliminates errors from connections that died
                while idle in the pool.

            ## Contributing

            1. Fork the repo and create a feature branch.
            2. Install dev dependencies: pip install -e ".[dev]"
            3. Run the test suite: pytest tests/ -v --tb=short
            4. Open a pull request with a clear description of the change.

            Please follow the existing code style (ruff for linting, black for
            formatting). All new features must include tests and updated docstrings.
            Bug fixes should include a regression test.

            ## License

            MIT. See LICENSE for full text.
        """),
        "key_facts": [
            "pool_size",
            "max_overflow",
            "pool_timeout",
            "pool_recycle",
            "pool_pre_ping",
            "pool_size + max_overflow",
            "SELECT 1",
        ],
    },

    # ── Case 2: Repetitive production log ───────────────────────────────────
    # Filler: ~35 routine INFO request lines with the same timestamp/status pattern.
    # Signal: the escalating WARN → ERROR → CRITICAL lines that reveal the cause.
    {
        "label": "Production log — memory crash diagnosis",
        "question": (
            "What caused the service to crash, and at what time did the first "
            "warning sign appear?"
        ),
        "context": textwrap.dedent("""\
            2025-06-10 03:30:01 INFO  [api] GET  /health 200 2ms
            2025-06-10 03:30:02 INFO  [api] POST /api/v1/ingest 201 18ms
            2025-06-10 03:30:02 INFO  [api] GET  /api/v1/reports/447 200 11ms
            2025-06-10 03:30:03 INFO  [api] GET  /api/v1/reports/448 200 9ms
            2025-06-10 03:30:03 INFO  [api] POST /api/v1/ingest 201 21ms
            2025-06-10 03:30:04 INFO  [api] GET  /api/v1/reports/449 200 10ms
            2025-06-10 03:30:04 INFO  [api] GET  /health 200 1ms
            2025-06-10 03:30:05 INFO  [api] POST /api/v1/ingest 201 17ms
            2025-06-10 03:30:05 INFO  [api] GET  /api/v1/reports/450 200 12ms
            2025-06-10 03:30:06 INFO  [api] GET  /api/v1/reports/451 200 8ms
            2025-06-10 03:30:06 INFO  [api] POST /api/v1/ingest 201 20ms
            2025-06-10 03:30:07 INFO  [api] GET  /health 200 2ms
            2025-06-10 03:30:07 INFO  [api] GET  /api/v1/reports/452 200 9ms
            2025-06-10 03:30:08 INFO  [api] POST /api/v1/ingest 201 19ms
            2025-06-10 03:30:08 INFO  [api] GET  /api/v1/reports/453 200 11ms
            2025-06-10 03:30:09 INFO  [api] GET  /api/v1/reports/454 200 10ms
            2025-06-10 03:30:09 INFO  [api] POST /api/v1/ingest 201 22ms
            2025-06-10 03:30:10 INFO  [api] GET  /health 200 1ms
            2025-06-10 03:30:10 INFO  [api] GET  /api/v1/reports/455 200 9ms
            2025-06-10 03:30:11 INFO  [api] POST /api/v1/ingest 201 18ms
            2025-06-10 03:30:11 INFO  [api] GET  /api/v1/reports/456 200 12ms
            2025-06-10 03:30:12 INFO  [api] GET  /api/v1/reports/457 200 8ms
            2025-06-10 03:30:12 INFO  [api] POST /api/v1/ingest 201 21ms
            2025-06-10 03:30:13 INFO  [api] GET  /health 200 2ms
            2025-06-10 03:30:13 INFO  [api] GET  /api/v1/reports/458 200 10ms
            2025-06-10 03:30:14 INFO  [api] POST /api/v1/ingest 201 19ms
            2025-06-10 03:30:14 INFO  [api] GET  /api/v1/reports/459 200 11ms
            2025-06-10 03:30:15 INFO  [api] GET  /api/v1/reports/460 200 9ms
            2025-06-10 03:30:15 INFO  [api] POST /api/v1/ingest 201 20ms
            2025-06-10 03:30:16 INFO  [api] GET  /health 200 1ms
            2025-06-10 03:30:16 INFO  [api] GET  /api/v1/reports/461 200 8ms
            2025-06-10 03:30:17 INFO  [api] POST /api/v1/ingest 201 17ms
            2025-06-10 03:30:17 INFO  [api] GET  /api/v1/reports/462 200 12ms
            2025-06-10 03:30:18 INFO  [api] GET  /api/v1/reports/463 200 10ms
            2025-06-10 03:30:18 INFO  [api] POST /api/v1/ingest 201 23ms
            2025-06-10 03:41:02 WARN  [worker] RSS memory 72% (2.88 GB / 4.00 GB) — approaching limit
            2025-06-10 03:41:18 WARN  [worker] RSS memory 81% (3.24 GB / 4.00 GB) — GC pressure detected; full GC triggered
            2025-06-10 03:41:35 WARN  [worker] GC pause 1240ms — throughput impact on report aggregation job
            2025-06-10 03:41:51 WARN  [worker] RSS memory 91% (3.64 GB / 4.00 GB) — GC running continuously
            2025-06-10 03:41:58 ERROR [worker] report aggregation job stalled: no progress in 15s under GC pressure
            2025-06-10 03:42:05 CRITICAL [worker] MemoryError: unable to allocate 1.8 GB for aggregation result buffer
            2025-06-10 03:42:05 CRITICAL [worker] Traceback:
            2025-06-10 03:42:05 CRITICAL [worker]   File "worker/aggregator.py", line 312, in _build_report
            2025-06-10 03:42:05 CRITICAL [worker]     result = numpy.zeros((n_rows, n_cols), dtype=numpy.float64)
            2025-06-10 03:42:05 CRITICAL [worker] MemoryError
            2025-06-10 03:42:06 CRITICAL [main] Unhandled exception in worker process — initiating shutdown
            2025-06-10 03:42:06 INFO  [main] Draining 142 in-flight requests
            2025-06-10 03:42:09 INFO  [main] Shutdown complete
        """),
        "key_facts": [
            "03:41:02",
            "RSS memory",
            "GC pressure",
            "MemoryError",
            "unable to allocate 1.8 GB",
            "aggregation result buffer",
            "numpy.zeros",
        ],
    },

    # ── Case 3: Incident post-mortem document ────────────────────────────────
    # Filler: executive summary, impact metrics, long minute-by-minute timeline.
    # Signal: the "Root Cause" and "Contributing Factors" sections are explicit
    #         and state the cause verbatim.
    {
        "label": "Incident post-mortem — root cause question",
        "question": (
            "What was the root cause of the incident, and what were the "
            "contributing factors listed in the post-mortem?"
        ),
        "context": textwrap.dedent("""\
            # Incident Post-Mortem: API Outage — 2025-05-14

            **Severity:** P1
            **Duration:** 47 minutes (03:08 UTC – 03:55 UTC)
            **Author:** Platform Reliability Team
            **Status:** Closed

            ## Executive Summary

            On 2025-05-14, the payment-service API was unavailable for 47 minutes
            affecting all checkout flows. The incident was triggered by a bad deploy
            that introduced an unbounded retry loop in the Stripe webhook handler.
            This caused worker thread exhaustion which cascaded into full service
            unavailability. The incident was resolved by rolling back the deploy.

            ## Impact

            - 47 minutes of complete checkout unavailability
            - Estimated 2,300 failed payment attempts
            - Estimated $180,000 in deferred revenue (payments completed after recovery)
            - 0 data loss or corrupted records
            - 4 customer escalations

            ## Timeline (all times UTC)

            03:08 — Deploy of payment-service v3.41.0 completed. Canary at 5%.
            03:09 — First webhook delivery failures appear in Stripe dashboard.
            03:11 — Error rate on /webhooks/stripe crosses 1% alert threshold.
            03:12 — On-call engineer (Jamie) paged. Acknowledges at 03:14.
            03:14 — Jamie checks dashboards. Error rate now 8%. Begins investigation.
            03:16 — Thread pool saturation alert fires: 98% of 200 worker threads busy.
            03:18 — Jamie suspects database slowness; checks DB metrics — all normal.
            03:21 — Thread dumps show all threads stuck in webhook retry loop.
            03:23 — Jamie identifies the v3.41.0 deploy as the change window.
            03:25 — Escalates to on-call lead (Priya). Rollback decision made.
            03:27 — Rollback of payment-service to v3.40.2 initiated.
            03:31 — Rollback complete. Worker threads begin recovering.
            03:38 — Error rate drops below 1%. Checkout flow confirmed functional.
            03:55 — Monitoring returns to baseline. Incident closed.

            ## Root Cause

            The root cause was a missing base-case check in the Stripe webhook retry
            handler introduced in v3.41.0. The handler called itself recursively
            without a maximum retry limit when Stripe returned a 429 rate-limit
            response. Under normal load this path was never exercised, but a brief
            spike in Stripe 429s (caused by a concurrent Stripe platform issue)
            triggered the recursive loop. Each webhook request spawned a new thread
            on retry, exhausting the 200-thread worker pool within 8 minutes and
            blocking all request processing.

            Specific code path: `StripeWebhookHandler._deliver()` in
            `payment_service/webhooks/stripe.py`, lines 88-102.

            ## Contributing Factors

            1. No retry limit on the webhook handler: the retry logic had no
               max_retries cap, making unbounded recursion possible.
            2. No thread pool circuit breaker: the service had no mechanism to
               shed load when the worker thread pool exceeded 90% utilisation.
            3. Canary stage too short: the canary was promoted to 100% after only
               3 minutes at 5% traffic, insufficient to detect the failure under
               realistic Stripe 429 rates.
            4. Staging environment does not simulate Stripe 429 responses: the
               bug would have been caught in staging if rate-limit responses were
               included in the integration test suite.

            ## Action Items

            | Action | Owner | Due |
            |--------|-------|-----|
            | Add max_retries=5 cap to StripeWebhookHandler._deliver() | Jamie | 2025-05-17 |
            | Implement thread pool circuit breaker (shed at 90% utilisation) | Priya | 2025-05-21 |
            | Extend canary stage minimum to 15 minutes | Platform Reliability | 2025-05-28 |
            | Add Stripe 429 simulation to integration test suite | Jamie | 2025-05-28 |
            | Review all other webhook handlers for similar patterns | Security & Reliability | 2025-06-04 |

            ## Lessons Learned

            Recursive retry logic without a depth limit is a known anti-pattern.
            Our existing code review checklist did not include a check for unbounded
            recursion in retry handlers. The canary promotion policy will be updated
            to require a minimum observation window before full rollout.
        """),
        "key_facts": [
            "missing base-case check",
            "recursive",
            "max_retries",
            "200-thread worker pool",
            "no thread pool circuit breaker",
            "canary stage too short",
            "Stripe 429",
        ],
    },

    # ── Case 4: Architecture Decision Record ─────────────────────────────────
    # Filler: background, requirements, three alternatives with pros/cons,
    #         open questions, consequences.
    # Signal: the "Decision" section explicitly states the choice and rationale.
    {
        "label": "Architecture Decision Record — caching strategy",
        "question": (
            "What caching strategy was chosen and what were the reasons "
            "for rejecting the alternatives?"
        ),
        "context": textwrap.dedent("""\
            # ADR-0047: Distributed Cache Strategy for the Recommendations Service

            **Date:** 2025-04-02
            **Status:** Accepted
            **Deciders:** Backend Platform, Data Engineering, SRE

            ## Context

            The recommendations service currently recomputes personalised product
            rankings on every request by querying the ML feature store and running
            a lightweight scoring model. At current traffic (12,000 req/s peak),
            this adds 180–220 ms of latency and accounts for 34% of feature-store
            read load. We need to cache recommendations to reduce latency to under
            50 ms and cut feature-store load by at least 60%.

            Recommendations are per-user and staleness of up to 5 minutes is
            acceptable per product requirements. The cache must survive individual
            node failures without a full cold-start.

            ## Requirements

            - P99 read latency ≤ 50 ms
            - Cache hit rate ≥ 90% during peak hours
            - Tolerate loss of a single cache node without full cold-start
            - Stale-while-revalidate semantics: serve stale data while refreshing
            - Invalidation on explicit user-preference change (within 10 seconds)
            - No vendor lock-in beyond what we already have

            ## Options Considered

            ### Option A: In-process LRU cache (per pod)

            Each service pod maintains its own LRU cache in memory.

            Pros:
            - Zero network latency; reads are sub-millisecond
            - No new infrastructure to operate
            - Simple to implement with functools.lru_cache or cachetools

            Cons:
            - Cache is not shared across pods; each of 40 pods warms independently,
              multiplying feature-store load by 40x on cold start
            - Memory pressure: each pod would need ~800 MB for full working set,
              exceeding current pod memory limits
            - Invalidation on preference change requires broadcasting to all 40 pods
              with no reliable delivery guarantee
            - No persistence: pod restart = full cold start for that pod

            ### Option B: Redis Cluster with read replicas

            A dedicated Redis Cluster (3 primary shards, 1 replica each) with keys
            structured as rec:{user_id}:{context_hash}.

            Pros:
            - Shared cache across all pods; one warm cache serves the fleet
            - Redis supports TTL natively; stale-while-revalidate via Lua scripts
            - Targeted key deletion for preference-change invalidation
            - Proven at our scale; we already operate Redis for session storage

            Cons:
            - Adds ~2–5 ms network round-trip per cache read (acceptable)
            - Requires cluster sizing and operational overhead
            - Hot-key risk if a small number of users generate disproportionate traffic

            ### Option C: CDN edge caching (Cloudflare Workers KV)

            Cache recommendations at the CDN edge, keyed by user ID in Cloudflare
            Workers KV.

            Pros:
            - Lowest possible latency for geographically distributed users
            - Infinitely scalable reads with no operational overhead
            - Invalidation API available

            Cons:
            - Workers KV eventual consistency model: invalidation can take up to
              60 seconds to propagate globally, violating our 10-second requirement
            - Recommendations contain personalised data; routing through Cloudflare
              raises data residency questions under GDPR for EU users
            - Significant new vendor dependency; KV pricing at our volume is
              estimated at $4,200/month vs $380/month for Redis

            ## Decision

            We will use **Option B: Redis Cluster with read replicas**.

            Redis meets all stated requirements and reuses infrastructure we already
            operate confidently. The 2–5 ms network overhead is well within the 50 ms
            latency budget. Stale-while-revalidate semantics will be implemented via
            a Lua script that atomically checks TTL, returns the cached value, and
            enqueues a background refresh when the TTL falls below a 60-second
            threshold. Targeted key deletion (DEL rec:{user_id}:*) will handle
            preference-change invalidation within seconds.

            Option A was rejected because the per-pod cache does not share state,
            cold-start amplification violates the feature-store load requirement,
            and reliable broadcast invalidation is impractical at 40 pods.

            Option C was rejected because Workers KV eventual consistency cannot
            meet the 10-second invalidation requirement, and the GDPR data residency
            risk for EU users is unacceptable without significant additional work.

            ## Consequences

            - SRE will provision a 3-shard Redis Cluster (r7g.large, 1 replica each)
              in the primary region; estimated $380/month.
            - Recommendations service will be updated to use the shared Redis client
              library (v2.4+) which already handles retry and circuit-breaking.
            - A Lua script for stale-while-revalidate will be reviewed by the
              Data Engineering team before deploy.
            - Cache warming strategy on deploy: pre-populate top 10,000 users by
              request volume from the feature store during the canary phase.

            ## Open Questions

            - Should we add a local in-process L1 cache (10-second TTL) in front of
              Redis to absorb hot-key traffic without full in-process cache drawbacks?
              Decision deferred to implementation phase.
            - Cross-region replication: out of scope for this ADR; revisit if we
              expand to a second AWS region.
        """),
        "key_facts": [
            "Redis Cluster",
            "Lua script",
            "stale-while-revalidate",
            "Option A was rejected",
            "cold-start amplification",
            "Option C was rejected",
            "Workers KV eventual consistency",
            "10-second invalidation",
        ],
    },

    # ── Case 5: Long production log (long context) ───────────────────────────
    # Filler: ~70 repetitive INFO/DEBUG payment lines (worker ready, heartbeat, etc.)
    # Signal: WARN→ERROR→CRITICAL Stripe failure cascade in the last 10 lines.
    {
        "label": "Long production log — payment incident (long context)",
        "question": (
            "What caused the payment processing failures and at what time did "
            "the first error appear?"
        ),
        "context": textwrap.dedent("""\
            2025-07-01 09:00:01 INFO  [payment-svc] worker-1 ready pid=44201
            2025-07-01 09:00:01 INFO  [payment-svc] worker-2 ready pid=44202
            2025-07-01 09:00:01 INFO  [payment-svc] worker-3 ready pid=44203
            2025-07-01 09:00:02 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:05 INFO  [payment-svc] processed txn_001 amount=12.50 status=ok
            2025-07-01 09:00:05 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:08 INFO  [payment-svc] processed txn_002 amount=99.00 status=ok
            2025-07-01 09:00:08 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:00:11 INFO  [payment-svc] processed txn_003 amount=45.00 status=ok
            2025-07-01 09:00:11 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:14 INFO  [payment-svc] processed txn_004 amount=200.00 status=ok
            2025-07-01 09:00:14 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:17 INFO  [payment-svc] processed txn_005 amount=15.75 status=ok
            2025-07-01 09:00:17 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:00:20 INFO  [payment-svc] processed txn_006 amount=32.00 status=ok
            2025-07-01 09:00:20 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:23 INFO  [payment-svc] processed txn_007 amount=8.99 status=ok
            2025-07-01 09:00:23 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:26 INFO  [payment-svc] processed txn_008 amount=150.00 status=ok
            2025-07-01 09:00:26 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:00:29 INFO  [payment-svc] processed txn_009 amount=67.50 status=ok
            2025-07-01 09:00:29 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:32 INFO  [payment-svc] processed txn_010 amount=22.00 status=ok
            2025-07-01 09:00:32 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:35 INFO  [payment-svc] processed txn_011 amount=88.00 status=ok
            2025-07-01 09:00:35 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:00:38 INFO  [payment-svc] processed txn_012 amount=5.00 status=ok
            2025-07-01 09:00:38 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:41 INFO  [payment-svc] processed txn_013 amount=310.00 status=ok
            2025-07-01 09:00:41 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:44 INFO  [payment-svc] processed txn_014 amount=75.00 status=ok
            2025-07-01 09:00:44 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:00:47 INFO  [payment-svc] processed txn_015 amount=42.00 status=ok
            2025-07-01 09:00:47 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:50 INFO  [payment-svc] processed txn_016 amount=19.99 status=ok
            2025-07-01 09:00:50 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:53 INFO  [payment-svc] processed txn_017 amount=130.00 status=ok
            2025-07-01 09:00:53 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:00:56 INFO  [payment-svc] processed txn_018 amount=55.00 status=ok
            2025-07-01 09:00:56 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:00:59 INFO  [payment-svc] processed txn_019 amount=90.00 status=ok
            2025-07-01 09:00:59 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:01:02 WARN  [payment-svc] Stripe API latency elevated: p99=1840ms (threshold=500ms)
            2025-07-01 09:01:05 INFO  [payment-svc] processed txn_020 amount=14.00 status=ok
            2025-07-01 09:01:05 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:01:08 INFO  [payment-svc] processed txn_021 amount=250.00 status=ok
            2025-07-01 09:01:08 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:01:11 INFO  [payment-svc] processed txn_022 amount=33.00 status=ok
            2025-07-01 09:01:11 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:01:14 INFO  [payment-svc] processed txn_023 amount=180.00 status=ok
            2025-07-01 09:01:14 DEBUG [payment-svc] heartbeat ok latency=3ms
            2025-07-01 09:01:17 INFO  [payment-svc] processed txn_024 amount=7.50 status=ok
            2025-07-01 09:01:17 DEBUG [payment-svc] heartbeat ok latency=2ms
            2025-07-01 09:01:18 ERROR [payment-svc] stripe_error=api_connection_error attempt=1/3 txn_025
            2025-07-01 09:01:19 ERROR [payment-svc] stripe_error=api_connection_error attempt=2/3 txn_025
            2025-07-01 09:01:20 ERROR [payment-svc] stripe_error=api_connection_error attempt=3/3 txn_025
            2025-07-01 09:01:20 CRITICAL [payment-svc] Stripe circuit breaker OPEN after 3 consecutive failures
            2025-07-01 09:01:20 ERROR [payment-svc] signature verification failed — possible clock skew detected (delta=+4800ms)
            2025-07-01 09:01:20 CRITICAL [payment-svc] all payment workers halted — manual intervention required
        """),
        "key_facts": [
            "09:01:02",
            "Stripe API latency elevated",
            "api_connection_error",
            "circuit breaker OPEN",
            "signature verification failed",
            "clock skew",
        ],
    },

    # ── Case 6: Long application settings file ───────────────────────────────
    # Filler: database, cache, email, storage, logging, celery sections.
    # Signal: the SESSION and AUTH_PASSWORD_VALIDATORS sections.
    {
        "label": "Django settings file — session and auth config",
        "question": (
            "How is session storage configured, what is the session timeout, "
            "and what password validation rules are enforced?"
        ),
        "context": textwrap.dedent("""\
            # ============================================================
            # settings.py — Production configuration
            # ============================================================

            import os
            from pathlib import Path

            BASE_DIR = Path(__file__).resolve().parent.parent
            SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]
            DEBUG = False
            ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",")

            # ── Installed apps ───────────────────────────────────────────
            INSTALLED_APPS = [
                "django.contrib.admin",
                "django.contrib.auth",
                "django.contrib.contenttypes",
                "django.contrib.sessions",
                "django.contrib.messages",
                "django.contrib.staticfiles",
                "rest_framework",
                "rest_framework_simplejwt",
                "corsheaders",
                "storages",
                "apps.users",
                "apps.orders",
                "apps.payments",
                "apps.notifications",
                "apps.reports",
            ]

            MIDDLEWARE = [
                "django.middleware.security.SecurityMiddleware",
                "whitenoise.middleware.WhiteNoiseMiddleware",
                "django.contrib.sessions.middleware.SessionMiddleware",
                "corsheaders.middleware.CorsMiddleware",
                "django.middleware.common.CommonMiddleware",
                "django.middleware.csrf.CsrfViewMiddleware",
                "django.contrib.auth.middleware.AuthenticationMiddleware",
                "django.contrib.messages.middleware.MessageMiddleware",
                "django.middleware.clickjacking.XFrameOptionsMiddleware",
            ]

            ROOT_URLCONF = "config.urls"
            WSGI_APPLICATION = "config.wsgi.application"

            # ── Database ─────────────────────────────────────────────────
            DATABASES = {
                "default": {
                    "ENGINE": "django.db.backends.postgresql",
                    "NAME": os.environ["DB_NAME"],
                    "USER": os.environ["DB_USER"],
                    "PASSWORD": os.environ["DB_PASSWORD"],
                    "HOST": os.environ.get("DB_HOST", "localhost"),
                    "PORT": os.environ.get("DB_PORT", "5432"),
                    "CONN_MAX_AGE": 60,
                    "OPTIONS": {
                        "connect_timeout": 10,
                        "sslmode": "require",
                    },
                }
            }

            # ── Cache ────────────────────────────────────────────────────
            CACHES = {
                "default": {
                    "BACKEND": "django_redis.cache.RedisCache",
                    "LOCATION": os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                    "OPTIONS": {
                        "CLIENT_CLASS": "django_redis.client.DefaultClient",
                        "SOCKET_CONNECT_TIMEOUT": 5,
                        "SOCKET_TIMEOUT": 5,
                        "IGNORE_EXCEPTIONS": True,
                    },
                    "KEY_PREFIX": "app",
                    "TIMEOUT": 300,
                }
            }

            # ── Email ────────────────────────────────────────────────────
            EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
            EMAIL_HOST = os.environ.get("SMTP_HOST", "smtp.sendgrid.net")
            EMAIL_PORT = 587
            EMAIL_USE_TLS = True
            EMAIL_HOST_USER = os.environ.get("SMTP_USER", "apikey")
            EMAIL_HOST_PASSWORD = os.environ["SENDGRID_API_KEY"]
            DEFAULT_FROM_EMAIL = "no-reply@example.com"
            SERVER_EMAIL = "ops@example.com"

            # ── File storage (S3) ────────────────────────────────────────
            DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
            STATICFILES_STORAGE = "storages.backends.s3boto3.S3StaticStorage"
            AWS_STORAGE_BUCKET_NAME = os.environ["S3_BUCKET"]
            AWS_S3_REGION_NAME = os.environ.get("AWS_REGION", "us-east-1")
            AWS_S3_FILE_OVERWRITE = False
            AWS_DEFAULT_ACL = None
            AWS_S3_OBJECT_PARAMETERS = {"CacheControl": "max-age=86400"}

            # ── Logging ──────────────────────────────────────────────────
            LOGGING = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "json": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"},
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "formatter": "json",
                    },
                },
                "root": {"handlers": ["console"], "level": "INFO"},
                "loggers": {
                    "django": {"handlers": ["console"], "level": "WARNING", "propagate": False},
                    "apps": {"handlers": ["console"], "level": "INFO", "propagate": False},
                },
            }

            # ── Celery ───────────────────────────────────────────────────
            CELERY_BROKER_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            CELERY_RESULT_BACKEND = CELERY_BROKER_URL
            CELERY_TASK_SERIALIZER = "json"
            CELERY_RESULT_SERIALIZER = "json"
            CELERY_ACCEPT_CONTENT = ["json"]
            CELERY_TIMEZONE = "UTC"
            CELERY_TASK_SOFT_TIME_LIMIT = 300
            CELERY_TASK_TIME_LIMIT = 360
            CELERY_WORKER_MAX_TASKS_PER_CHILD = 200

            # ── Session ──────────────────────────────────────────────────
            SESSION_ENGINE = "django.contrib.sessions.backends.cache"
            SESSION_CACHE_ALIAS = "default"
            SESSION_COOKIE_AGE = 86400          # 24 hours in seconds
            SESSION_COOKIE_SECURE = True        # HTTPS only
            SESSION_COOKIE_HTTPONLY = True      # not accessible via JavaScript
            SESSION_COOKIE_SAMESITE = "Lax"
            SESSION_EXPIRE_AT_BROWSER_CLOSE = False
            SESSION_SAVE_EVERY_REQUEST = False  # only save on modification

            # ── Password validation ──────────────────────────────────────
            AUTH_PASSWORD_VALIDATORS = [
                {
                    "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
                },
                {
                    "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
                    "OPTIONS": {"min_length": 12},
                },
                {
                    "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
                },
                {
                    "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
                },
            ]

            # ── REST Framework ───────────────────────────────────────────
            REST_FRAMEWORK = {
                "DEFAULT_AUTHENTICATION_CLASSES": [
                    "rest_framework_simplejwt.authentication.JWTAuthentication",
                ],
                "DEFAULT_PERMISSION_CLASSES": [
                    "rest_framework.permissions.IsAuthenticated",
                ],
                "DEFAULT_THROTTLE_CLASSES": [
                    "rest_framework.throttling.AnonRateThrottle",
                    "rest_framework.throttling.UserRateThrottle",
                ],
                "DEFAULT_THROTTLE_RATES": {"anon": "100/hour", "user": "2000/hour"},
                "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
                "PAGE_SIZE": 50,
            }
        """),
        "key_facts": [
            "SESSION_ENGINE",
            "django.contrib.sessions.backends.cache",
            "SESSION_COOKIE_AGE",
            "86400",
            "MinimumLengthValidator",
            "min_length",
            "12",
            "NumericPasswordValidator",
        ],
    },
]


# ── Core logic ────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    label: str
    question: str
    key_facts: list[str]
    context_tokens_full: int
    context_tokens_compressed: int
    answer_full: str
    answer_compressed: str
    score_full: int          # 1-10
    score_compressed: int    # 1-10
    facts_in_full: list[str]
    facts_in_compressed: list[str]
    judge_reasoning: str
    compression_ratio: float  # Claude input tokens: compressed / full
    text_ratio: float         # imptokens: n_kept / n_original
    param: float              # threshold (logprob) or target_reduction (sentence)
    compression_latency_ms: float
    api_time_full_ms: float
    api_time_compressed_ms: float
    mode: str
    compressed_context: str = field(repr=False)

    @property
    def threshold(self) -> float:
        """Backwards-compatible alias for param."""
        return self.param


def compress(
    text: str,
    param: float,
    binary: str,
    mode: str = "sentence",
    query: str = "",
) -> tuple[str, int, int, float, float]:
    """
    Run imptokens compression.

    Returns (compressed_text, n_kept, n_original, text_ratio, latency_ms).
    """
    if mode == "sentence":
        cmd = [binary, "--file", "-", "--sentence-mode",
               f"--target-reduction={param}", "--output-format", "json"]
        if query:
            cmd.append(f"--query={query}")
    else:
        cmd = [binary, "--file", "-", f"--threshold={param}", "--output-format", "json"]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=120)
    latency_ms = (time.perf_counter() - t0) * 1000

    if result.returncode != 0:
        raise RuntimeError(f"imptokens failed:\n{result.stderr[:400]}")
    data = json.loads(result.stdout)
    return (
        data["compressed_text"],
        data["n_kept"],
        data["n_original"],
        data["compression_ratio"],
        latency_ms,
    )


def ask_claude(client: anthropic.Anthropic, model: str,
               question: str, context: str) -> tuple[str, int]:
    """Send question + context to Claude, return (answer, input_tokens)."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text, resp.usage.input_tokens


def judge(client: anthropic.Anthropic, model: str,
          question: str, key_facts: list[str],
          answer_a: str, answer_b: str) -> tuple[int, int, list[str], list[str], str]:
    """
    Ask Claude to rigorously score two answers against known key facts.
    Blind — the judge doesn't know which answer came from compressed context.
    Returns (score_a, score_b, facts_in_a, facts_in_b, reasoning).
    """
    facts_fmt = "\n".join(f"  - {f}" for f in key_facts)
    prompt = textwrap.dedent(f"""\
        You are a strict technical evaluator. Two AI assistants answered the same question.
        Your job is to score both answers rigorously using the rubric below.

        Question: {question}

        Key facts a correct, complete answer MUST address:
        {facts_fmt}

        Answer A:
        {answer_a}

        Answer B:
        {answer_b}

        Scoring rubric (1–10):
          10 = all key facts addressed, no errors, actionable
           8 = most key facts present, at most one minor omission, no errors
           6 = several key facts missing OR explanation is vague but not wrong
           4 = key facts mostly absent OR contains a factual error
           2 = wrong answer or irrelevant to the question
           1 = completely off-topic or refuses to answer

        Instructions:
        1. For each answer, list exactly which key facts from the list above it mentions.
           Use the exact strings from the list. If a fact is absent, do not list it.
        2. Assign a score using the rubric. Be strict — missing a key fact costs points.
        3. Write one sentence comparing the two answers.

        Respond in this exact JSON (no extra text, no markdown fences):
        {{"facts_in_a": ["<exact fact string>", ...], "facts_in_b": ["<exact fact string>", ...], "score_a": <1-10>, "score_b": <1-10>, "reasoning": "<one sentence>"}}
    """)
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    return (
        int(data["score_a"]),
        int(data["score_b"]),
        data.get("facts_in_a", []),
        data.get("facts_in_b", []),
        data["reasoning"],
    )


def run_case(client: anthropic.Anthropic, case: dict, model: str,
             param: float, binary: str, mode: str = "threshold") -> CaseResult:
    label = case["label"]
    question = case["question"]
    context = case["context"]
    key_facts = case["key_facts"]

    param_desc = f"target_reduction={param}" if mode == "sentence" else f"threshold={param}"
    print(f"  Compressing context ({param_desc})…", end=" ", flush=True)
    query = question if mode == "sentence" else ""
    compressed_context, n_kept, n_original, text_ratio, compression_latency_ms = compress(
        context, param, binary, mode=mode, query=query
    )
    saved_pct = (1 - text_ratio) * 100
    print(f"✓  {n_original}→{n_kept} tokens ({saved_pct:.0f}% dropped, {compression_latency_ms:.0f}ms)")

    print(f"  Asking Claude (full context)…", end=" ", flush=True)
    t0 = time.perf_counter()
    answer_full, input_tokens_full = ask_claude(client, model, question, context)
    api_time_full_ms = (time.perf_counter() - t0) * 1000
    print(f"✓ ({input_tokens_full} tokens, {api_time_full_ms:.0f}ms)")
    time.sleep(0.5)

    print(f"  Asking Claude (compressed)…", end=" ", flush=True)
    t0 = time.perf_counter()
    answer_compressed, input_tokens_compressed = ask_claude(
        client, model, question, compressed_context
    )
    api_time_compressed_ms = (time.perf_counter() - t0) * 1000
    print(f"✓ ({input_tokens_compressed} tokens, {api_time_compressed_ms:.0f}ms)")
    time.sleep(0.5)

    print(f"  Judging answers against {len(key_facts)} key facts…", end=" ", flush=True)
    score_full, score_compressed, facts_in_full, facts_in_compressed, reasoning = judge(
        client, model, question, key_facts, answer_full, answer_compressed
    )
    full_cov = f"{len(facts_in_full)}/{len(key_facts)}"
    comp_cov = f"{len(facts_in_compressed)}/{len(key_facts)}"
    print(f"✓  full={score_full}/10 ({full_cov} facts)  compressed={score_compressed}/10 ({comp_cov} facts)")

    compression_ratio = input_tokens_compressed / input_tokens_full if input_tokens_full else 1.0

    return CaseResult(
        label=label,
        question=question,
        key_facts=key_facts,
        context_tokens_full=input_tokens_full,
        context_tokens_compressed=input_tokens_compressed,
        answer_full=answer_full,
        answer_compressed=answer_compressed,
        score_full=score_full,
        score_compressed=score_compressed,
        facts_in_full=facts_in_full,
        facts_in_compressed=facts_in_compressed,
        judge_reasoning=reasoning,
        compression_ratio=compression_ratio,
        text_ratio=text_ratio,
        param=param,
        compression_latency_ms=compression_latency_ms,
        api_time_full_ms=api_time_full_ms,
        api_time_compressed_ms=api_time_compressed_ms,
        mode=mode,
        compressed_context=compressed_context,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

WIDTH = 72

def bar(score: int, width: int = 10) -> str:
    filled = round(score / 10 * width)
    return "█" * filled + "░" * (width - filled)

def verdict(full: int, comp: int) -> str:
    delta = comp - full
    if delta >= 0:
        return "✓ equivalent or better"
    if delta == -1:
        return "~ negligible drop"
    return f"✗ degraded ({delta:+d})"


def quality_verdict(delta: float, saved_pct: float) -> tuple[str, bool, bool]:
    """
    Savings-aware verdict. Returns (label, is_pass, is_marginal).

    Rationale: a -1.5 delta is acceptable when you saved 60% of tokens, but
    unacceptable when you saved only 5%. The tolerance scales with savings:
      pass tolerance     = saved_pct * 0.030  (e.g. 50% → ±1.5)
      marginal tolerance = saved_pct * 0.050  (e.g. 50% → ±2.5)
    """
    pass_tol     = saved_pct * 0.030
    marginal_tol = saved_pct * 0.050
    if delta >= -pass_tol:
        return "PASS ✓", True, False
    if delta >= -marginal_tol:
        return "MARGINAL ~", False, True
    return "FAIL ✗", False, False

def wrap_indent(text: str, width: int = 60, indent: str = "    ") -> str:
    lines = text.strip().splitlines()
    out = []
    for line in lines:
        wrapped = textwrap.fill(line or " ", width=width)
        for wl in wrapped.splitlines():
            out.append(indent + wl)
    return "\n".join(out[:8])  # cap at 8 lines for display


def fact_coverage(facts_seen: list[str], facts_all: list[str]) -> str:
    n = len(facts_seen)
    d = len(facts_all)
    pct = 100 * n / d if d else 0
    return f"{n}/{d} ({pct:.0f}%)"


def _param_label(mode: str, param: float) -> str:
    if mode == "sentence":
        return f"target_reduction={param}"
    return f"threshold={param}"


def print_report(results: list[CaseResult], model: str, param: float, mode: str):
    W = "═" * WIDTH
    D = "─" * WIDTH

    print(f"\n{W}")
    print(f"  imptokens × Claude — Answer Quality Benchmark")
    print(f"  Model: {model}  |  {len(results)} cases  |  mode: {mode}  |  {_param_label(mode, param)}")
    print(W)

    total_full = sum(r.context_tokens_full for r in results)
    total_comp = sum(r.context_tokens_compressed for r in results)
    total_saved = total_full - total_comp

    for i, r in enumerate(results, 1):
        saved = r.context_tokens_full - r.context_tokens_compressed
        saved_pct = (1 - r.compression_ratio) * 100
        text_saved_pct = (1 - r.text_ratio) * 100
        v = verdict(r.score_full, r.score_compressed)
        missing_facts = [f for f in r.key_facts if f not in r.facts_in_compressed]

        print(f"\n  [{i}/{len(results)}] {r.label}")
        print(f"  {D}")
        print(f"  Text compression:  {text_saved_pct:.0f}% tokens dropped  "
              f"(latency: {r.compression_latency_ms:.0f}ms)")
        print(f"  Claude tokens:     {r.context_tokens_full:,} → {r.context_tokens_compressed:,}"
              f"  ({saved_pct:.0f}% saved, {saved:,} tokens)")
        print(f"  Quality:           Full {r.score_full}/10 {bar(r.score_full)}"
              f"   Compressed {r.score_compressed}/10 {bar(r.score_compressed)}   {v}")
        print(f"  Key fact coverage: Full {fact_coverage(r.facts_in_full, r.key_facts)}"
              f"   Compressed {fact_coverage(r.facts_in_compressed, r.key_facts)}")
        if missing_facts:
            print(f"  Missing facts:     {', '.join(missing_facts)}")
        print(f"  Judge:             \"{r.judge_reasoning}\"")
        print()
        print(f"  Full answer:")
        print(wrap_indent(r.answer_full))
        print(f"\n  Compressed answer:")
        print(wrap_indent(r.answer_compressed))

    avg_full = sum(r.score_full for r in results) / len(results)
    avg_comp = sum(r.score_compressed for r in results) / len(results)
    avg_saved_pct = (1 - total_comp / total_full) * 100 if total_full else 0
    avg_delta = avg_comp - avg_full
    avg_facts_full = sum(len(r.facts_in_full) / len(r.key_facts) for r in results if r.key_facts) / len(results)
    avg_facts_comp = sum(len(r.facts_in_compressed) / len(r.key_facts) for r in results if r.key_facts) / len(results)
    avg_compress_ms   = sum(r.compression_latency_ms for r in results) / len(results)
    avg_api_full_ms   = sum(r.api_time_full_ms for r in results) / len(results)
    avg_api_comp_ms   = sum(r.api_time_compressed_ms for r in results) / len(results)
    api_delta_ms      = avg_api_full_ms - avg_api_comp_ms
    net_overhead_ms   = avg_compress_ms - api_delta_ms

    print(f"\n{W}")
    print(f"  SUMMARY  ({_param_label(mode, param)}, mode={mode})")
    print(D)
    print(f"  Total tokens sent (full):       {total_full:,}")
    print(f"  Total tokens sent (compressed): {total_comp:,}")
    print(f"  Total tokens saved:             {total_saved:,}  ({avg_saved_pct:.1f}%)")
    print(D)
    print(f"  Avg answer quality — full:      {avg_full:.1f}/10")
    print(f"  Avg answer quality — compressed:{avg_comp:.1f}/10")
    print(f"  Quality delta:                  {avg_delta:+.1f}/10  "
          f"({abs(avg_delta)/avg_full*100:.1f}% {'loss' if avg_delta < 0 else 'gain'})")
    print(D)
    print(f"  Avg key-fact coverage — full:      {avg_facts_full*100:.0f}%")
    print(f"  Avg key-fact coverage — compressed:{avg_facts_comp*100:.0f}%")
    print(D)
    print(f"  Latency breakdown (avg per case):")
    print(f"    Compression (local, imptokens): {avg_compress_ms:>7.0f}ms  ← overhead added")
    print(f"    API call — full context:        {avg_api_full_ms:>7.0f}ms")
    print(f"    API call — compressed:          {avg_api_comp_ms:>7.0f}ms  ({api_delta_ms:+.0f}ms vs full)")
    net_str = f"{net_overhead_ms:+.0f}ms net" if net_overhead_ms > 0 else f"{abs(net_overhead_ms):.0f}ms net saved"
    print(f"    Net wall-clock change:          {net_str}")
    print(D)

    v_label, is_pass, is_marginal = quality_verdict(avg_delta, avg_saved_pct)
    if is_pass:
        verdict_str = f"✓ PASS — quality loss within tolerance for {avg_saved_pct:.0f}% savings"
    elif is_marginal:
        verdict_str = f"~ MARGINAL — acceptable tradeoff for {avg_saved_pct:.0f}% savings"
    else:
        verdict_str = f"✗ FAIL — quality loss too large relative to {avg_saved_pct:.0f}% savings"

    print(f"  Verdict: {verdict_str}")
    print(W)


def save_markdown(results: list[CaseResult], path: str, model: str, param: float, mode: str):
    lines = [
        f"# imptokens × Claude — Quality Benchmark\n",
        f"**Model:** `{model}` | **Mode:** `{mode}` | **{_param_label(mode, param)}** | **Cases:** {len(results)}\n",
        "## Results\n",
        "| Case | Claude tokens (full→compressed) | Saved | Latency | Score full | Score compressed | Facts full | Facts compressed | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        saved_pct = (1 - r.compression_ratio) * 100
        v = verdict(r.score_full, r.score_compressed)
        lines.append(
            f"| {r.label} | {r.context_tokens_full:,} → {r.context_tokens_compressed:,} "
            f"| {saved_pct:.0f}% | {r.compression_latency_ms:.0f}ms "
            f"| {r.score_full}/10 | {r.score_compressed}/10 "
            f"| {fact_coverage(r.facts_in_full, r.key_facts)} "
            f"| {fact_coverage(r.facts_in_compressed, r.key_facts)} | {v} |"
        )

    total_full = sum(r.context_tokens_full for r in results)
    total_comp = sum(r.context_tokens_compressed for r in results)
    avg_full = sum(r.score_full for r in results) / len(results)
    avg_comp = sum(r.score_compressed for r in results) / len(results)
    avg_saved = (1 - total_comp / total_full) * 100
    avg_facts_full = sum(len(r.facts_in_full) / len(r.key_facts) for r in results if r.key_facts) / len(results)
    avg_facts_comp = sum(len(r.facts_in_compressed) / len(r.key_facts) for r in results if r.key_facts) / len(results)
    avg_latency = sum(r.compression_latency_ms for r in results) / len(results)

    lines += [
        f"\n## Summary\n",
        f"- **Mode:** `{results[0].mode}`",
        f"- **Total tokens saved:** {total_full - total_comp:,} ({avg_saved:.1f}%)",
        f"- **Average quality — full context:** {avg_full:.1f}/10",
        f"- **Average quality — compressed:** {avg_comp:.1f}/10",
        f"- **Quality delta:** {avg_comp - avg_full:+.1f}/10",
        f"- **Avg key-fact coverage — full:** {avg_facts_full*100:.0f}%",
        f"- **Avg key-fact coverage — compressed:** {avg_facts_comp*100:.0f}%",
        f"- **Avg compression latency:** {avg_latency:.0f}ms\n",
        "## Per-case answers\n",
    ]
    for r in results:
        missing = [f for f in r.key_facts if f not in r.facts_in_compressed]
        lines += [
            f"### {r.label}\n",
            f"**Question:** {r.question}\n",
            f"**Key facts:** {', '.join(f'`{f}`' for f in r.key_facts)}\n",
            f"**Compression latency:** {r.compression_latency_ms:.0f}ms\n",
            f"**Full answer ({r.context_tokens_full:,} tokens sent):**\n",
            f"> {r.answer_full.strip()}\n",
            f"**Compressed answer ({r.context_tokens_compressed:,} tokens sent):**\n",
            f"> {r.answer_compressed.strip()}\n",
            f"**Facts covered (compressed):** {', '.join(f'`{f}`' for f in r.facts_in_compressed) or 'none'}\n",
            f"**Missing facts:** {', '.join(f'`{f}`' for f in missing) or 'none'}\n",
            f"**Judge:** {r.judge_reasoning}\n",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _sweep_row(results: list[CaseResult]) -> tuple[float, float, float, float, float, float]:
    """Compute (saved_pct, avg_full, avg_comp, delta, avg_fact_cov, avg_latency)."""
    total_full = sum(r.context_tokens_full for r in results)
    total_comp = sum(r.context_tokens_compressed for r in results)
    avg_full   = sum(r.score_full for r in results) / len(results)
    avg_comp   = sum(r.score_compressed for r in results) / len(results)
    saved_pct  = (1 - total_comp / total_full) * 100 if total_full else 0
    delta      = avg_comp - avg_full
    avg_fact_cov = sum(
        len(r.facts_in_compressed) / len(r.key_facts) for r in results if r.key_facts
    ) / len(results)
    avg_latency = sum(r.compression_latency_ms for r in results) / len(results)
    return saved_pct, avg_full, avg_comp, delta, avg_fact_cov, avg_latency


def print_sweep_report(sweep: dict[float, list[CaseResult]], model: str, mode: str):
    """Print a parameter sweep analysis table and identify the optimal cutoff."""
    W = "═" * WIDTH
    param_name = "Reduction" if mode == "sentence" else "Threshold"

    print(f"\n{W}")
    print(f"  imptokens × Claude — {param_name} Sweep  [mode: {mode}]")
    print(f"  Model: {model}  |  {len(sweep)} values tested")
    print(W)

    header = (f"  {param_name:>10}  │  {'Saved':>6}  │  {'Facts%':>6}  │"
              f"  {'Q(full)':>7}  │  {'Q(comp)':>7}  │  {'Delta':>6}  │  {'Latency':>8}  │  Verdict")
    sep = "  " + "─" * (len(header) - 2)
    print(header)
    print(sep)

    optimal_param: Optional[float] = None
    sweep_rows = []

    # For sentence mode: sort ascending (least → most aggressive: 0.05 → 0.45)
    # For logprob mode: sort descending (least → most aggressive: -0.05 → -0.5)
    sort_keys = sorted(sweep.keys(), reverse=(mode != "sentence"))
    for param in sort_keys:
        results = sweep[param]
        if not results:
            continue
        saved_pct, avg_full, avg_comp, delta, avg_fact_cov, avg_latency = _sweep_row(results)

        v, is_pass, _ = quality_verdict(delta, saved_pct)
        if is_pass:
            optimal_param = param  # most aggressive PASS as we sweep

        param_str = f"{param:.4g}"
        print(f"  {param_str:>10}  │  {saved_pct:>5.1f}%  │  {avg_fact_cov*100:>5.0f}%  │"
              f"  {avg_full:>7.1f}  │  {avg_comp:>7.1f}  │  {delta:>+6.1f}  │"
              f"  {avg_latency:>6.0f}ms  │  {v}")
        sweep_rows.append((param, saved_pct, avg_full, avg_comp, delta, v, avg_fact_cov, avg_latency))

    print(sep)
    print(f"  (Tolerance: PASS if quality drop ≤ saved% × 0.03 pts; MARGINAL if ≤ saved% × 0.05 pts)")

    if optimal_param is not None:
        row = next(r for r in sweep_rows if r[0] == optimal_param)
        print(f"\n  Optimal {param_name.lower()}: {optimal_param:.4g} "
              f"({row[1]:.1f}% savings, {row[6]*100:.0f}% fact retention, "
              f"{row[7]:.0f}ms avg latency, quality within tolerance)")
    else:
        print(f"\n  No value passed. Try a less aggressive setting.")
    print(W)


def save_sweep_markdown(sweep: dict[float, list[CaseResult]], path: str, model: str, mode: str):
    param_name = "Reduction" if mode == "sentence" else "Threshold"
    sort_keys = sorted(sweep.keys()) if mode == "sentence" else sorted(sweep.keys(), reverse=True)
    lines = [
        f"# imptokens × Claude — {param_name} Sweep\n",
        f"**Model:** `{model}` | **Mode:** `{mode}` | "
        f"**Values tested:** {', '.join(str(t) for t in sort_keys)}\n",
        f"## {param_name} Comparison\n",
        f"| {param_name} | Tokens Saved | Fact coverage | Avg Quality (full) | Avg Quality (comp) | Delta | Avg Latency | Verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    optimal_param: Optional[float] = None
    for param in sort_keys:
        results = sweep[param]
        if not results:
            continue
        saved_pct, avg_full, avg_comp, delta, avg_fact_cov, avg_latency = _sweep_row(results)

        v, is_pass, _ = quality_verdict(delta, saved_pct)
        if is_pass:
            optimal_param = param

        lines.append(
            f"| {param} | {saved_pct:.1f}% | {avg_fact_cov*100:.0f}% "
            f"| {avg_full:.1f}/10 | {avg_comp:.1f}/10 | {delta:+.1f} "
            f"| {avg_latency:.0f}ms | {v} |"
        )

    lines.append(f"\n> Tolerance: PASS if quality drop ≤ saved% × 0.03 pts; MARGINAL if ≤ saved% × 0.05 pts\n")
    if optimal_param is not None:
        lines.append(f"\n**Optimal {param_name.lower()}:** `{optimal_param:.4g}` (most aggressive with PASS verdict)\n")
    else:
        lines.append(f"\n**No value passed.** Try a less aggressive setting.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSweep report saved → {path}")


def main():
    ap = argparse.ArgumentParser(
        description="imptokens × Claude quality benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Sentence mode (default) — no model required, fast compression:
              --target-reduction 0.05   very light  ~5% tokens removed
              --target-reduction 0.10   light       ~10% tokens removed  (default)
              --target-reduction 0.20   moderate    ~20% tokens removed
              --target-reduction 0.45   significant ~45% tokens removed

            Logprob mode — requires llama.cpp model:
              --threshold -0.05   very light   ~10-20% tokens dropped
              --threshold -0.10   light        ~30-40% tokens dropped
              --threshold -0.20   moderate     ~40-50% tokens dropped
              --threshold -0.50   aggressive   ~50-65% tokens dropped
        """),
    )
    ap.add_argument("--mode", choices=["sentence", "logprob"], default="sentence",
                    help="Compression mode (default: sentence)")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Claude model to use (default: claude-haiku-4-5-20251001)")
    # Sentence mode params
    ap.add_argument("--target-reduction", type=float, default=0.1,
                    help="Sentence mode: fraction to remove (default: 0.1)")
    ap.add_argument("--target-reductions", type=float, nargs="+", metavar="R",
                    help="Sentence mode sweep: e.g. --target-reductions 0.05 0.1 0.2 0.45")
    # Logprob mode params
    ap.add_argument("--threshold", type=float, default=-0.1,
                    help="Logprob mode: logprob threshold (default: -0.1)")
    ap.add_argument("--thresholds", type=float, nargs="+", metavar="T",
                    help="Logprob mode sweep: e.g. --thresholds -0.05 -0.1 -0.2 -0.3 -0.5")
    ap.add_argument("--save-report", metavar="PATH",
                    help="Save full markdown report to this file")
    ap.add_argument("--cases", type=int, default=None,
                    help="Run only the first N cases (for quick testing)")
    ap.add_argument("--binary", default=None, metavar="PATH",
                    help="Path to imptokens binary (default: auto-detect on PATH or ./target/release/imptokens)")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY environment variable first.")

    binary = args.binary or shutil.which("imptokens") or "./target/release/imptokens"
    if not shutil.which(binary) and not os.path.isfile(binary):
        sys.exit(f"imptokens binary not found. Build with: cargo build --release")

    client = anthropic.Anthropic(api_key=api_key)
    cases = CASES[:args.cases] if args.cases else CASES
    mode = args.mode

    # Resolve sweep list and single param for the chosen mode.
    if mode == "sentence":
        sweep_params = sorted(set(args.target_reductions)) if args.target_reductions else None
        single_param = args.target_reduction
    else:
        sweep_params = sorted(set(args.thresholds), reverse=True) if args.thresholds else None
        single_param = args.threshold

    # ── Sweep mode ─────────────────────────────────────────────────────────────
    if sweep_params:
        print(f"imptokens × Claude — parameter sweep  [mode: {mode}]")
        print(f"Model: {args.model}  |  values: {sweep_params}  |  {len(cases)} cases each")
        print(f"Binary: {binary}\n")

        sweep: dict[float, list[CaseResult]] = {}
        for param in sweep_params:
            print(f"\n{'━' * WIDTH}")
            print(f"  {_param_label(mode, param)}")
            print(f"{'━' * WIDTH}")
            param_results: list[CaseResult] = []
            for i, case in enumerate(cases, 1):
                print(f"\n[{i}/{len(cases)}] {case['label']}")
                try:
                    result = run_case(client, case, args.model, param, binary, mode)
                    param_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                print()
            sweep[param] = param_results

        print_sweep_report(sweep, args.model, mode)

        if args.save_report:
            save_sweep_markdown(sweep, args.save_report, args.model, mode)
        return

    # ── Single-param mode ──────────────────────────────────────────────────────
    print(f"imptokens × Claude — Quality Benchmark  [mode: {mode}]")
    print(f"Model: {args.model}  |  {_param_label(mode, single_param)}  |  {len(cases)} cases")
    print(f"Binary: {binary}\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['label']}")
        try:
            result = run_case(client, case, args.model, single_param, binary, mode)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        print()

    if not results:
        sys.exit("No results — all cases failed.")

    print_report(results, args.model, single_param, mode)

    if args.save_report:
        save_markdown(results, args.save_report, args.model, single_param, mode)


if __name__ == "__main__":
    main()
