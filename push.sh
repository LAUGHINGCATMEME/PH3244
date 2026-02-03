#!/bin/sh

set -e

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Error: not inside a git repository."
    exit 1
}

find . -type f \( \
    -name "*.pdf" -o \
    -name "*.png" -o \
    -name "*.md" -o \
    -name "*.jpg" -o \
    -name "*.tex" -o \
    -name "*.bib" -o \
    -name "*.sh" \
\) -print0 | xargs -0 git add --

printf "Commit message: "
IFS= read -r msg

[ -z "$msg" ] && {
    echo "Error: empty commit message."
    exit 1
}

git commit -m "$msg"
git push

