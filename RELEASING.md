# Releasing

Releases are managed through GitHub Actions. **Do not tag releases manually.**
A manually pushed `v*` tag will not produce a proper release with changelog
and PyPI publication.

Go to **Actions → Release → Run workflow** and fill in the inputs:

| Input | Default | Notes |
| --- | --- | --- |
| `increment` | `none` | Leave on `none` to let Commitizen decide from the commit history. Override with `PATCH`, `MINOR`, or `MAJOR` to force the bump size. |
| `prerelease` | `none` | Add a prerelease suffix: `alpha`, `beta`, or `rc`. When set, the post-release `devN` bump is skipped. |
| `devrelease` | `false` | See below. |
| `manual_version` | *(empty)* | Pin an exact version (e.g. `3.0.0`), bypassing Commitizen's increment logic. |

If no releasable commits exist since the last tag (e.g. only `chore` or `docs`
commits), Commitizen exits without bumping, the workflow completes successfully,
and no release artefact is produced. Use `increment` to force a bump in that
situation.

## Production release (`devrelease: false`)

The standard release flow. Commitizen analyses commits since the last tag,
bumps the version, and creates the bump commit and an annotated tag with the
incremental changelog as the tag message. The changes are pushed to `main` via
an automatically created and merged pull request (to satisfy branch protection
rules). The package is then built and published to PyPI, a GitHub Release is
created with the changelog and build artifacts attached, and a follow-up `devN`
commit is pushed to `main` so the in-tree version never matches a published
release.

## Development release (`devrelease: true`)

Use this when you need a version available on PyPI for testing before it is
ready to become the official production release. Commitizen appends a `.devN`
suffix to the bumped version, so both the tag and the published artefact carry
it (e.g. `2.1.0.dev0`). No post-release `devN` bump is applied afterwards.
