# Contributing

Thank you for contributing to MATCHA. Please read this guide before opening an
issue or pull request.

## Commit messages

This project uses [Commitizen](https://commitizen-tools.github.io/commitizen/)
with the conventional-commit format. Commitizen reads the commit history to
determine the next version number and to generate the changelog automatically,
so well-formed commit messages are important.

The format is:

```text
<type>[(<scope>)]: <subject>

[optional body]

[optional footer]
```

Scope is optional. Common types: `feat`, `fix`, `docs`, `refactor`, `test`,
`ci`, `build`, `chore`. A `feat` triggers a MINOR bump; a `fix` triggers a
PATCH bump; a footer containing `BREAKING CHANGE:` triggers a MAJOR bump.

| Type | Bump |
| ---- | ---- |
| `feat` | MINOR |
| `fix` | PATCH |
| `BREAKING CHANGE` footer | MAJOR |
| `docs`, `ci`, `chore`, `refactor`, `test`, `build` | No release |

## Development setup

1. Install [mise](https://mise.jdx.dev/getting-started.html) and activate it in your shell.
2. Run `mise install` in the repo root to install pinned tool versions.
3. Install Python dependencies: `uv sync`.
4. Pre-commit hooks install automatically via the mise hook on directory entry
   (or run `pre-commit install` manually).

## Pull requests

- Branch off `main` and open your PR against `main`.
- One logical change per PR makes review faster.
- All CI checks and unit tests must pass before merge.
- If your branch ends up behind main, first rebase onto main before merging to keep git history more readable

## Testing

Run the test suite with:

```bash
uv run python -m pytest -k 'not gpu'
```

To measure coverage locally, run pytest under `coverage` and print a report:

```bash
uv run coverage run -m pytest -k 'not gpu'
uv run coverage report
```

Coverage is enforced in CI via [Codecov](https://codecov.io/gh/emdgroup/matcha).
Each PR gets two coverage checks alongside the tests badge:

- **project** — fails if total coverage drops by more than 1%.
- **patch** — targets 80% coverage on the lines the PR changes (5% tolerance).

Codecov also posts a comment on each PR with the coverage delta and a per-file
breakdown of touched files. Configuration lives in `codecov.yml` (thresholds,
ignored paths) and `[tool.coverage.*]` in `pyproject.toml` (measurement scope,
line exclusions).

## Releasing a new version

Releases are handled by a maintainer via GitHub Actions.
**Do not tag releases manually.**
A manually pushed `v*` tag will not produce a proper release with changelog
and PyPI publication.

## LLMs and AI agents policy

Purely LLM-generated contributions are strictly forbidden.

LLMs are allowed for writing code, debugging, drafting issues and PRs and so forth, but human oversight is required.

For typical development workflows, we strongly recommend the amazing [mach10](https://github.com/LeanAndMean/mach10) to structure issues and PRs.

## Repo structure and code design

Please check out the CLAUDE.md file to get an overview of the general coding guidelines and structure of the package.
