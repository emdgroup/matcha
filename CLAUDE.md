# Repo guide

Orientation for contributors (human or AI). Keep this file short and honest — if the layout changes, update it.

For the `src/matcha` internals, see the CLAUDE.md inside that folder (coming next). This file only covers the top level.

## What this is

MATCHA is a Python toolkit for molecular property prediction. It wraps descriptor, SMILES and graph-based models behind one scikit-learn-style API and a small CLI. Published to PyPI as `emd-matcha`.

## Top-level layout

```text
matcha/
├── src/matcha/          # the package — see src/matcha/CLAUDE.md
├── tests/               # pytest suite, folder-per-subpackage mirror of src/matcha/
├── skills/              # Claude Code skills scoped to this repo (not shipped in the wheel)
├── docs/                # Docs for the package
├── .github/             # CI workflows, custom actions, dependabot, zizmor config
├── PATTERNS.md          # repo-wide architectural patterns (registries, schemas, managers, ...)
├── pyproject.toml       # single source of truth for deps, extras, tooling config
├── uv.lock              # locked resolution — commit changes when deps move
├── .mise.toml           # pins Python and tool versions (uv, pre-commit)
├── .pre-commit-config.yaml
├── CONTRIBUTING.md      # commit format, PR flow, dev setup
├── RELEASING.md         # release procedure (maintainers only)
└── README.md
```

## How the pieces fit

- **`src/matcha/` is the product.** Everything else exists to build, test, document, or release it. The wheel packages only `src/matcha` (see `[tool.hatch.build.targets.wheel]` in `pyproject.toml`). Repo-wide architectural patterns (registries, schemas, managers, base+concrete, ...) live in `PATTERNS.md`; per-subpackage details in each subpackage's `CLAUDE.md`.
- **`tests/` mirrors `src/matcha/`** subpackage-for-subpackage. New module in `src/matcha/foo/bar.py` → new test file in `tests/foo/test_bar.py`.
- **`skills/matcha-data-scientist/`** is a Claude Code skill for people using MATCHA as data scientists (not for developing MATCHA itself). It is not published with the wheel.
- **`.github/workflows/`** runs tests, pre-commit, docs, PR checks and release. `release.yml` is driven by Commitizen and must not be triggered by pushing tags manually — see `RELEASING.md`.
- **`docs/`** contains the wiki of the package, and some recipes for extending the codebase (e.g. adding a new model or featurization option).

## Conventions worth knowing up front

- **Python 3.12** (`>=3.12,<3.14`). Match the version in `.mise.toml` — don't upgrade casually.
- **uv** for dependency management, **mise** for tool pinning. Run `mise install` then `uv sync`; pre-commit hooks install on first `cd` in.
- **Ruff** (line length 88, target py312) and **Pyright** (basic mode, `src/` only) are the linters. Config lives in `pyproject.toml`.
- **Pytest** runs with `filterwarnings = ["error", ...]` — new warnings will fail CI. If a warning is expected, add it to the ignore list rather than silencing at the call site.
- **Coverage** is measured with `coverage` (config in `pyproject.toml`, scope = `src/matcha`, branch coverage on). CI uploads to Codecov and fails the PR if project coverage drops >1% or patch coverage falls below 80%. Locally: `uv run coverage run -m pytest -k 'not gpu' && uv run coverage report`.
- **CPU vs GPU torch** are declared as conflicting extras (`[tool.uv]` `conflicts`). Install one or the other, never both.
- **Commits use Commitizen / conventional commits.** The commit type drives the next release version — see `CONTRIBUTING.md` for the mapping.

## Gotchas

- Editing `pyproject.toml` deps? Run `uv lock` and commit `uv.lock` in the same change.
- The GPU test suite is skipped by default; run `uv run pytest -k 'not gpu'` locally, same as CI.
- Don't push `v*` tags by hand. Releases go through the GitHub Actions workflow.
