#!/usr/bin/env bash
# Fetch the four source figures of Devlin, Chang, Lee & Toutanova (2018),
# "BERT: Pre-training of Deep Bidirectional Transformers for Language
# Understanding" (arXiv:1810.04805v2), and convert them into assets for the
# BERT chapters of this visualization.
#
# The PDFs are inputs only and are never committed. See LICENSE-BERT-figures.md
# for the attribution terms.
#
# Why the pipeline is PDF -> ghostscript -> SVG rather than a single pdftocairo call:
# `pdftocairo -svg` writes text as SVG 1.1 font/glyph references (<use
# xlink:href="#glyph0-1">) and embeds no font, and Firefox renders none of it -- the
# figure arrives with its labels missing. Ghostscript's -dNoOutputFonts replaces every
# glyph with its outline first, so the SVG depends on no font at all. Verified on
# fig2-input: 195 glyph references -> 0, 158 paths -> 295. A per-file PNG fallback
# remains for the case where either tool is unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/public/bert-figs"
SRC_URL="https://arxiv.org/e-print/1810.04805v2"

mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "downloading $SRC_URL"
curl -sSL --retry 3 -o "$TMP/bert.e-print" "$SRC_URL"
tar xzf "$TMP/bert.e-print" -C "$TMP"

failed=0

# convert_one <source pdf> <output stem>
#
# Verified against poppler 25.03 / ghostscript 10.02: `pdftocairo -svg <in> <base>` writes
# the SVG to exactly <base> -- no extension, no page suffix -- for a single-page figure, and
# rejects -singlefile for -svg output. Page count is asserted rather than assumed, so a figure
# that one day ships as two pages fails loudly here instead of silently contributing half a
# drawing. Conversion errors are printed rather than swallowed: otherwise a systematic mistake
# (a bad flag, a missing binary) would surface as four quiet PNG downgrades.
convert_one() {
  local pdf="$1" stem="$2"
  local svg="$OUT/$stem.svg" outlined="$TMP/$stem-outlined.pdf"
  local err size
  err=$(mktemp)

  if [ ! -f "$TMP/$pdf" ]; then
    echo "  !! $pdf not present in the arXiv source" >&2
    rm -f "$err"; failed=$((failed + 1)); return 0
  fi

  local pages
  pages=$(pdfinfo "$TMP/$pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
  if [ "$pages" != "1" ]; then
    echo "  !! $pdf has ${pages:-unknown} pages; expected a single-page figure" >&2
    rm -f "$err"; failed=$((failed + 1)); return 0
  fi

  if gs -q -dNoOutputFonts -sDEVICE=pdfwrite -o "$outlined" "$TMP/$pdf" 2>"$err" \
     && pdftocairo -svg "$outlined" "$OUT/$stem" 2>>"$err"; then
    if [ -f "$OUT/$stem" ]; then
      mv "$OUT/$stem" "$svg"
    fi
    size=$(stat -c%s "$svg" 2>/dev/null || echo 0)
    if [ "$size" -gt 2048 ] && ! grep -q 'href="#glyph' "$svg"; then
      echo "  ok  $pdf -> $(basename "$svg") ($size bytes, text outlined)"
      rm -f "$err"; return 0
    fi
    if [ "$size" -le 2048 ]; then
      echo "  ~~  $pdf -> SVG only $size bytes (too small to be the figure); using PNG fallback"
    else
      echo "  ~~  $pdf -> SVG still contains glyph references (text would be missing in Firefox); using PNG fallback"
    fi
  else
    echo "  ~~  conversion failed for $pdf: $(head -c 300 "$err" | tr '\n' ' ')" >&2
  fi

  rm -f "$svg" "$OUT/$stem" "$err"
  if pdftoppm -png -r 300 -singlefile "$TMP/$pdf" "$OUT/$stem" 2>/dev/null && [ -f "$OUT/$stem.png" ]; then
    size=$(stat -c%s "$OUT/$stem.png")
    echo "  ok  $pdf -> $stem.png ($size bytes) [PNG fallback -- will not scale as sharply]"
    return 0
  fi
  echo "  !! $pdf: both SVG and PNG conversion failed" >&2
  failed=$((failed + 1))
}

convert_one "BERT_Overall.pdf"      "fig1-overview"
convert_one "Input_Emebeddings.pdf" "fig2-input"
convert_one "BERT_comparisons.pdf"  "fig3-comparison"
convert_one "BERT_fine_tune.pdf"    "fig4-fine-tune"

echo
echo "public/bert-figs:"
ls -la "$OUT"

if [ "$failed" -ne 0 ]; then
  echo "$failed figure(s) could not be produced" >&2
  exit 1
fi
