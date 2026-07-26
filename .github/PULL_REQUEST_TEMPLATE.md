## Summary

<!-- What does this PR change, and why? Link any issue. -->

## Checklist

- [ ] `ruff check` + `ruff format --check` pass
- [ ] `pytest --cov-fail-under=60` passes (coverage not lowered)
- [ ] `bandit -r src/ -c pyproject.toml` — no new HIGH/CRITICAL
- [ ] Tests added for any new behavior (mock-based, no API key)
- [ ] **Zero `koboi.*` imports** added (decoupled extraction)
- [ ] Docs updated (CHANGELOG / README / docstrings) if user-facing

## Notes

<!-- Anything reviewers should know: breaking changes, follow-ups, risks. -->
