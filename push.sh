#!/bin/sh

set -e

# Ensure inside git repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Error: not inside a git repository."
    exit 1
}

echo "=== Compiling LaTeX files ==="

for i in 0 1 2 3 4 5 6 7 8 9; do
    texfile="./$i/$i.tex"

    if [ -f "$texfile" ]; then
        echo "Compiling $texfile"
        (cd "$i" && latexmk --pdf "$i.tex")
    else
        echo "Warning: $texfile not found, skipping"
    fi
done

echo "=== Cleaning extra files (excluding .git) ==="

# Delete everything NOT in allowed extensions, excluding .git directory
find . -type f \
  -not -path "./.git/*" \
  \( \
    -name "*.aux" -o \
    -name "*.bbl" -o \
    -name "*.blg" -o \
    -name "*.fdb*" -o \
    -name "*.fls" -o \
    -name "*.log" -o \
    -name "*.toc" \
  \) \
  -print -delete

echo "=== Staging files ==="

git add .

printf "Commit message: "
IFS= read -r msg

[ -z "$msg" ] && {
    echo "Error: empty commit message."
    exit 1
}

git commit -m "$msg"
git push

echo "=== Done ==="
