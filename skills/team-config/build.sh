#!/bin/sh
# Assemble a standalone HTML blueprint from team YAML and the bundled template.
# Usage: sh build.sh <team.yaml> [output.html]
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SRC=${1:-}
OUT=${2:-team-blueprint.html}

if [ -z "$SRC" ]; then
  echo "usage: sh build.sh <team.yaml> [output.html]" >&2
  exit 2
fi
if [ ! -f "$SRC" ]; then
  echo "build.sh: team file not found: $SRC" >&2
  exit 1
fi
if [ -e "$OUT" ] && [ "$SRC" -ef "$OUT" ]; then
  echo "build.sh: input and output refer to the same file" >&2
  exit 2
fi
for part in blueprint.head.html blueprint.body.html blueprint.foot.html vendor/js-yaml.min.js vendor/js-yaml.LICENSE; do
  if [ ! -f "$HERE/$part" ]; then
    echo "build.sh: missing template part: $HERE/$part" >&2
    exit 1
  fi
done

# Encode YAML for the raw-text script island. Escaping every '<' prevents HTML
# script end-tags and parser-state changes; escaping '&' first keeps the
# transport reversible, including literal entities and backslashes in prompts.
# The renderer decodes these two substitutions once before parsing YAML.
guard() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' "$1"; }

# Busybox ash can ignore group-redirection failures under set -e, so the
# explicit failure branch below prevents reporting an unwritable output as success.
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
{
  cat "$HERE/blueprint.head.html"
  guard "$SRC"
  cat "$HERE/blueprint.body.html"
  printf '%s\n' '/* js-yaml license notice'
  cat "$HERE/vendor/js-yaml.LICENSE"
  printf '%s\n' '*/'
  cat "$HERE/vendor/js-yaml.min.js"
  cat "$HERE/blueprint.foot.html"
} > "$OUT" || { echo "build.sh: cannot write output: $OUT" >&2; exit 1; }

echo "wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes) — open it in a browser."
