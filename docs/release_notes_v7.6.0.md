# Release notes (since v7.5.0)

## Highlights
- Added collateral-agnostic performance metrics (`adg_pnl`, `mdg_pnl`, PnL-based Sharpe/Sortino, with weighted variants) and documented them.
- Improved coin overrides UX and safety with new docs, tests, and debug logging.
- Refactored position/balance handling to separate updates and optimize Hyperliquid’s combined fetch.
- Robustified data-sync, fill-events cache layout, and Kucoin open-order pagination.
- Expanded optimizer limit handling with a new `limit_utils` module and tighter limit/scoring integration.

## Metrics & Analysis
- New metrics: `adg_pnl`, `mdg_pnl` (mean/median daily PnL over end-of-day balance), and PnL-based `sharpe_ratio_pnl`/`sortino_ratio_pnl`, plus weighted versions.
- Added `docs/metrics.md` entries clarifying equity vs PnL Sharpe/Sortino semantics.
- Types/analysis wiring updated across Rust and Python consumers; optimization weights and config canonicalization recognize the new metrics.
- Added optimizer-side limit utilities (`limit_utils`) and integration tests to enforce limit normalization/penalties consistently.

## Coin Overrides
- Documented full override behavior (`docs/coin_overrides.md`) with inline/file-based examples, allowed fields, path resolution, and pitfalls.
- Added debug logs when overrides are initialized and when override values are used.
- Expanded tests to cover path resolution, missing override files, and retention through config formatting.

## Exchange & Sync Fixes
- Kucoin: `fetch_open_orders` now paginates (no more 50-order cap).
- Hyperliquid: `fill_events_manager` now supports the exchange; fetch positions/balance combined once per cycle.
- General: position/balance fetching split for all exchanges (dedicated `fetch_balance`/`fetch_positions`), with balance caching to avoid double-hits on combined endpoints.
- `sync_tar.py`: pulling a single file no longer nests an extra directory.
- Optimizer: new limit handling pipeline (`limit_utils`), enhanced scoring/limits parsing, and additional tests for optimizer limits.

## Fill Events
- Cache root now `caches/{exchange}/{user}` (was `caches/{exchange}_{user}`).
- Hyperliquid fetcher path registered; CLI no longer rejects hyperliquid users.

## Core Refactors & Safeguards
- `update_positions` and new `update_balance` separated; main loops call both.
- Balance caching reuse and rate-limit/network guards added.
- Tests added for balance split and Hyperliquid combined fetch reuse.
- Added optimizer limit integration tests and helper coverage for limit utilities.
- Fixed false-positive "stale" Rust extension detection when the rebuilt binary is identical to the local copy (mtime now synced on SHA match).

## Docs
- Added coin override guide and metrics reference updates.

## QA/Tests
- New tests: override path handling, balance split behavior, Hyperliquid fetch reuse, Hyperliquid fetcher wiring.
- Optimizer limit integration and limit_utils tests added.
- Existing suite continues to pass on targeted runs (spot-checked new tests).
