# Blast Radius — Per-Test Dynamic Coverage + AST Symbol Selector

Builds exact per-test line coverage (`sys.monitoring`), file-read maps (`open()`), and static AST statement/symbol/dialect-family dependency maps over `sqlglot` (`eb11aec4be8cb2122afe713a4816662c3f0fb2fa`) during `docker build`, then selects minimal candidate tests per patch and schedules them with optimal LPT bin-packing across 1..8 workers.

## Open-Source Dependencies

- `sqlglot` (pinned commit `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`) and packages from `requirements.lock` (`pytest`, `duckdb`, `pandas`, `numpy`, etc.).
