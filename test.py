"""Generate CoReFusion fused images at their original spatial resolution."""

import argparse

from corefusion.inference import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pth")
    parser.add_argument("--ir", default="data/MSRS/test/ir")
    parser.add_argument("--vis", default="data/MSRS/test/vis")
    parser.add_argument("--output", default="results/MSRS")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every registered pair")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.checkpoint, args.ir, args.vis, args.output, args.device, args.patch_batch,
        args.seed, args.limit, args.overwrite)


if __name__ == "__main__":
    main()
