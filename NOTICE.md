# Attribution and release status

This package is a cleaned export of the author-selected D1 implementation for **CoReFusion: One-Step Infrared and Visible Image Fusion via Decoupled Reconstruction**.

The continuous-time shortcut objective in `corefusion/objective.py` is adapted from **ExplicitShortCut**, specifically its `loss/esc_loss.py` implementation. The retained source describes the CoReFusion adaptation: the empirical endpoint distribution uses the registered infrared/visible source pair. This attribution is retained when consolidating the original code into the `corefusion` package.

The MSRS, M3FD, and FMB example images belong to their respective dataset owners and retain their original filenames and pixels. Their inclusion does not transfer ownership or grant additional dataset rights. Dataset use and redistribution remain subject to the respective providers' terms.

PyTorch, NumPy, OpenCV, and optional evaluation dependencies retain their own licenses. Metric implementations retain any attribution present in their source files.

No software license or copyright holder was specified for this export. This package therefore does not assign an MIT, Apache, or other license to the CoReFusion code or checkpoint. The authors should select a release license and verify applicable third-party redistribution terms before publishing the repository.
