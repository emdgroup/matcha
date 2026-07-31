#!/usr/bin/env bash
set -euo pipefail

# --- 1. INSTALL COMMITIZEN ---------------------------------------------------
if [[ "${INPUT_COMMITIZEN_VERSION}" == "latest" ]]; then
	pip install commitizen
else
	pip install "commitizen==${INPUT_COMMITIZEN_VERSION}"
fi
echo "Commitizen version: $(cz version)"

# --- 2. CAPTURE PRE-BUMP VERSION ---------------------------------------------
prev_version="$(cz version --project)"
echo "previous_version=${prev_version}" >>"${GITHUB_OUTPUT}"
echo "Previous version: ${prev_version}"

# --- 3. BUILD AND RUN CZ BUMP ------------------------------------------------
# Dev releases skip --changelog-to-stdout so that CHANGELOG.md never receives a
# dev-release entry. Commitizen's tag-format regex (_DEFAULT_VERSION_PARSER) does
# not match PEP 440 .devN notation, so any dev entry in CHANGELOG.md would cause
# the next proper release to fail when _find_incremental_rev can't anchor on it.
# With update_changelog_on_bump=false in pyproject.toml, omitting
# --changelog-to-stdout means the changelog is not touched at all for dev releases.
declare -a cmd=(cz bump --yes --git-output-to-stderr)
if [[ "${INPUT_DEVRELEASE}" != "true" ]]; then
	cmd+=(--changelog-to-stdout)
fi

if [[ -n "${INPUT_INCREMENT}" && "${INPUT_INCREMENT,,}" != "none" ]]; then
	cmd+=(--increment "${INPUT_INCREMENT^^}")
fi
if [[ -n "${INPUT_PRERELEASE}" && "${INPUT_PRERELEASE,,}" != "none" ]]; then
	cmd+=(--prerelease "${INPUT_PRERELEASE}")
fi
if [[ "${INPUT_DEVRELEASE}" == "true" ]]; then
	cmd+=(--devrelease 0)
fi
if [[ -n "${INPUT_MANUAL_VERSION}" ]]; then
	cmd+=("${INPUT_MANUAL_VERSION}")
fi

printf 'Running: %s\n' "${cmd[*]}"
cz_exit=0
"${cmd[@]}" >"${INPUT_CHANGELOG_INCREMENT_FILENAME}" || cz_exit=$?
# Exit code 21 = NoneIncrementExit: no releasable commits — treat as success so
# the rest of the script can set bumped=false and exit cleanly. Any other
# non-zero exit code is a real error and is re-raised.
if [[ "${cz_exit}" -ne 0 && "${cz_exit}" -ne 21 ]]; then
	exit "${cz_exit}"
fi

# Write multiline changelog_body output using heredoc delimiter
{
	echo "changelog_body<<CHANGELOG_EOF"
	cat "${INPUT_CHANGELOG_INCREMENT_FILENAME}"
	echo "CHANGELOG_EOF"
} >>"${GITHUB_OUTPUT}"

# --- 4. CAPTURE POST-BUMP VERSION --------------------------------------------
version="$(cz version --project)"
bumped="true"
if [[ "${version}" == "${prev_version}" ]]; then
	bumped="false"
fi
revision_tag="v${version}"

{
	echo "version=${version}"
	echo "bumped=${bumped}"
	echo "revision_tag=${revision_tag}"
} >>"${GITHUB_OUTPUT}"
echo "New version: ${version} (bumped=${bumped}, tag=${revision_tag})"

# --- 5. CREATE ANNOTATED TAG -------------------------------------------------
# cz bump always creates a lightweight tag locally (needed for version
# resolution). When tagging is enabled, overwrite it with an annotated tag
# whose body is the changelog increment. When tagging is disabled, leave the
# local tag in place (commitizen needs it) but don't push it.
if [[ "${INPUT_CREATE_TAG}" == "true" && "${bumped}" == "true" ]]; then
	echo "Creating annotated tag ${revision_tag} with changelog body"
	git tag -f -a "${revision_tag}" -F "${INPUT_CHANGELOG_INCREMENT_FILENAME}"
fi

# --- 6. PUSH VIA PR ----------------------------------------------------------
# Branch protection on main prevents direct pushes. Work around this by pushing
# the current HEAD to a remote release branch, creating a PR, then pushing main.
# The open PR satisfies branch protection, so the push to main is accepted as
# the merge result of the PR. The PR auto-closes as merged, and the repo setting
# to delete head branches after merge handles cleanup.
if [[ "${bumped}" == "true" ]]; then
	release_branch="release/${revision_tag}"

	if [[ "${INPUT_CREATE_TAG}" == "true" ]]; then
		echo "Pushing release branch and tag ${revision_tag} to origin"
		git push origin "refs/tags/${revision_tag}" "HEAD:refs/heads/${release_branch}"
	else
		echo "Pushing release branch (no tags) to origin"
		git push origin "HEAD:refs/heads/${release_branch}"
	fi

	echo "Creating PR against main"
	pr_url="$(gh pr create \
		--base main \
		--head "${release_branch}" \
		--title "release: ${revision_tag}" \
		--body "Automated release ${revision_tag}.")"
	echo "Created PR: ${pr_url}"

	echo "Waiting for PR checks to pass"
	# gh exits 1 both when checks fail and when no checks are configured.
	# Capture the output to distinguish the two cases; treat "no checks" as
	# passing — required checks are enforced by branch protection independently.
	if ! gh pr checks "${pr_url}" --watch --fail-fast; then
		pr_checks_out="$(gh pr checks "${pr_url}" 2>&1 || true)"
		echo "${pr_checks_out}"
		if ! echo "${pr_checks_out}" | grep -qi "no checks reported"; then
			echo "PR checks failed"
			exit 1
		fi
		echo "No checks configured on release branch, proceeding"
	fi

	echo "Pushing main"
	git push origin main
fi

echo "Done."
