"""Read a photo of a handwritten answer.

Usage:
  python scripts/read_handwriting.py photo.jpg                       # transcribe (downloads TrOCR on first use, ~1.3 GB)
  python scripts/read_handwriting.py photo.jpg --save-lines /tmp/l   # also save each cut line as an image
  python scripts/read_handwriting.py photo.jpg --segment-only --save-lines /tmp/l   # no model: check the line cutting

Always look at the saved lines first on a new kind of photo: if the cutting is wrong, the reading will be too.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from gradient.ocr import TrOCRReader, doubtful_line_numbers, segment_lines  # noqa: E402


def main(argv=None, reader=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--save-lines", default=None, help="folder to save the cut line images")
    ap.add_argument("--segment-only", action="store_true", help="only cut lines, do not run the OCR model")
    ap.add_argument("--model", default=os.getenv("GRADIENT_OCR_MODEL", "microsoft/trocr-base-handwritten"))
    args = ap.parse_args(argv)

    image = Image.open(args.image)
    crops = segment_lines(image)
    print(f"{len(crops)} line(s) found")
    if args.save_lines:
        out = Path(args.save_lines)
        out.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(crops, 1):
            c.image.save(out / f"line_{i:02d}.png")
        print("saved line images to", out)
    if args.segment_only:
        return
    lines = (reader or TrOCRReader(args.model)).read(image)
    for i, l in enumerate(lines, 1):
        conf = "n/a" if l.confidence is None else f"{l.confidence:.2f}"
        print(f"{i:2d} [{conf}]{' CHECK' if l.doubtful else ''} {l.text}")
    flagged = doubtful_line_numbers(lines)
    if flagged:
        print("Please check line(s):", ", ".join(map(str, flagged)))


if __name__ == "__main__":
    main()
