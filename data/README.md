# Example data

Each split contains the first five registered infrared/visible pairs in lexicographic filename-stem order from the supplied dataset directories. Original filenames and image pixels are retained. `manifest.json` records each image's dimensions and SHA-256 checksum.

| Dataset | Split | Pairs | Files |
| --- | --- | ---: | ---: |
| MSRS | `train` | 5 | 10 |
| MSRS | `test` | 5 | 10 |
| M3FD | `test` | 5 | 10 |
| FMB | `test` | 5 | 10 |
| Total | | 20 | 40 |

The layout is `data/<dataset>/<split>/ir/` for infrared and `data/<dataset>/<split>/vis/` for visible images. Pairing uses exact filename stems; extensions may differ. Missing counterparts and duplicate stems are rejected.

The MSRS training loader deterministically reserves one of the five training pairs for validation and uses the other four for crop sampling. Full training data uses the validation fraction configured in `configs/corefusion_msrs.json`.

These files are small functional examples, not the full experimental datasets. Download full datasets from their providers for training or evaluation, then supply their directories through `--train-ir`/`--train-vis` or `--ir`/`--vis`. Full-dataset results cannot be inferred from the example subsets.

The data remain the property of the MSRS, M3FD, and FMB dataset owners; see [NOTICE.md](../NOTICE.md).
