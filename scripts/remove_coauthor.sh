#!/usr/bin/env bash
# Remove "Co-Authored-By: Claude ..." trailers from ALL commit messages on main, then force-push.
# RUN THIS YOURSELF (history rewrite + force-push). A backup branch is created first; contents are verified unchanged.
# Undo (before or after pushing, if needed):  git reset --hard backup-before-coauthor-removal-2  &&  git push --force-with-lease origin main
set -euo pipefail
cd "$(dirname "$0")/.."
git diff --quiet && git diff --cached --quiet || { echo "Commit or stash your changes first."; exit 1; }
git fetch origin
git branch backup-before-coauthor-removal-2
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --msg-filter 'grep -iv "^Co-Authored-By: Claude" | sed -e :a -e "/^\n*\$/{\$d;N;ba" -e "}"' -- main
left=$(git log --format=%B | grep -ci "co-authored-by: claude" || true)
echo "Claude trailers left: $left (expected 0)"
if ! git diff --quiet backup-before-coauthor-removal-2 main; then echo "ERROR: file contents differ from backup; NOT pushing."; exit 1; fi
echo "Contents identical to backup. Pushing with lease..."
git push --force-with-lease origin main
echo "Done."
