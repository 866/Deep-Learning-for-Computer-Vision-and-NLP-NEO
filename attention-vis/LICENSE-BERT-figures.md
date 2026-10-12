# License note: BERT paper figures in `public/bert-figs/`

The four images in `public/bert-figs/` are **not** this project's original work and are
**not** covered by the MIT license that covers this project's source code (see
`LICENSE-llm-viz` and the `"license": "MIT"` field of `package.json`, both of which apply to
the code). They are reproductions of figures from:

> Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. **BERT: Pre-training
> of Deep Bidirectional Transformers for Language Understanding.** In *Proceedings of the 2019
> Conference of the North American Chapter of the Association for Computational Linguistics:
> Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis,
> Minnesota. Association for Computational Linguistics. DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423).
> Preprint: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805). (NAACL 2019, Best Long Paper.)

## Files and their source

Produced by `scripts/get-bert-figures.sh` from the figure PDFs inside the arXiv e-print source
archive for `1810.04805v2`. The intermediate PDFs are inputs only and are not committed.

| file in `public/bert-figs/` | source file in the e-print | paper figure |
| --- | --- | --- |
| `fig1-overview.svg` | `BERT_Overall.pdf` | Figure 1 — overall model pre-training/fine-tuning architecture |
| `fig2-input.svg` | `Input_Emebeddings.pdf` | Figure 2 — token, segment, and position input embeddings |
| `fig3-comparison.svg` | `BERT_comparisons.pdf` | Figure 3 — BERT vs. OpenAI GPT and feature-based approaches |
| `fig4-fine-tune.svg` | `BERT_fine_tune.pdf` | Figure 4 — fine-tuning task architectures (incl. 4b, the single-sentence classification task this project reproduces) |

Conversion runs the source PDF through ghostscript `-dNoOutputFonts` (replacing each glyph with
its outline) and then `pdftocairo -svg`. The result is a *derived vector rendering* of the
original artwork: the drawing is the authors', the file format is ours. If ghostscript or
poppler is unavailable the script falls back to a 300 dpi PNG of the same source figure.

## How they are used here

Each figure is displayed in the BERT chapters of this interactive walkthrough, next to the
commentary that explains it, captioned with the paper title, authors, year and arXiv id, and
linked back to the paper. They are included to explain and comment on the published method —
the same figures the walkthrough's own 3D model reconstructs from real trained weights — in an
educational context, with full attribution and no suggestion of authorship by this project.

## What the rights situation actually is (checked, not assumed)

Checked on 2026-10-12:

- The published NAACL-HLT 2019 PDF carries the footer "*© 2019 Association for Computational
  Linguistics*" and **no** open-content notice on the paper itself
  (`https://aclanthology.org/N19-1423.pdf`, page 1).
- The arXiv e-print we converted from is listed under arXiv's
  [perpetual non-exclusive distribution license 1.0](https://arxiv.org/licenses/nonexclusive-distrib/1.0/),
  which authorises *arXiv* to distribute the work. It is not a reuse grant to third parties, and
  the authors did not select a CC option for this submission.
- The ACL Anthology site displays a CC BY 4.0 notice site-wide; that covers the Anthology's own
  text and metadata, and should not be read as a licence over a 2019 paper's figures.

So there is **no open-content grant** on these figures, and this project does not claim one.
This project is MIT-licensed, but that licence covers the code only. Anyone redistributing this
repository beyond the use described above — a fork that republishes the images elsewhere, a
commercial product, a derivative figure — should evaluate the position for themselves first, or
regenerate the visuals from scratch (the walkthrough's 3D rendering of the same architecture is
generated from this project's own code and weights, and is unencumbered).

This file records a documented, deliberate decision about what is shipped and why. It is a
attribution notice, not legal advice.
