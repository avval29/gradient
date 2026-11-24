"""Handwritten answer photo -> text.

Pipeline: split the page photo into text lines (classic projection profile on a local-contrast ink mask),
read each line with TrOCR, and report a per-line confidence so a teacher can check doubtful lines.

Human in the loop: the transcription is shown to the teacher to correct BEFORE it is graded, so an OCR
mistake never silently lowers a student's mark.

Known limits (be honest about these):
- TrOCR is trained on single lines of English handwriting (IAM); it will be weaker on cramped, slanted or
  very untidy writing, and on other scripts.
- Line splitting assumes dark ink on lighter paper, a roughly straight page and clear gaps between lines.
  Touching lines, heavy ruled lines, strong skew and shadows can break it. Use scripts/read_handwriting.py
  with --save-lines to see exactly what it cut.
- LOW_CONFIDENCE is an untuned starting point.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

LOW_CONFIDENCE = 0.5  # mean token probability below this -> ask the teacher to check the line
MIN_CONTRAST = 30     # a page with no pixel this much darker than its surroundings is treated as blank


@dataclass
class LineCrop:
    top: int
    bottom: int
    image: Image.Image


@dataclass
class OCRLine:
    text: str
    confidence: float | None
    top: int
    bottom: int

    @property
    def doubtful(self) -> bool:
        return self.confidence is not None and self.confidence < LOW_CONFIDENCE


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on a 1-D array of non-negative values."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.where(w0 == 0, 1, w0)
    total_mean = (hist * centers).sum() / hist.sum()
    m1 = (total_mean * hist.sum() - np.cumsum(hist * centers)) / np.where(w1 == 0, 1, w1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def ink_mask(gray_image: Image.Image) -> np.ndarray:
    """True where there is ink: pixels clearly darker than their local surroundings (robust to shadows).
    Returns an all-False mask for a blank page."""
    w = gray_image.width
    background = gray_image.filter(ImageFilter.BoxBlur(max(15, w // 40)))
    diff = np.clip(np.asarray(background, dtype=float) - np.asarray(gray_image, dtype=float), 0, None)
    if diff.max() < MIN_CONTRAST:
        return np.zeros(diff.shape, dtype=bool)
    return diff > max(otsu_threshold(diff[diff > 2]) if (diff > 2).any() else 0, MIN_CONTRAST / 2)


def _runs(active: np.ndarray) -> list[list[int]]:
    runs, start = [], None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(active)])
    return runs


def segment_lines(image: Image.Image, max_width: int = 1600, min_height_frac: float = 0.012,
                  merge_gap_frac: float = 0.006, min_ink_frac: float = 0.01) -> list[LineCrop]:
    """Cut a page photo into text-line crops, top to bottom."""
    img = image.convert("L")
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)))
    mask = ink_mask(img)
    h, w = mask.shape
    if not mask.any():
        return []
    k = max(1, h // 200)
    profile = np.convolve(mask.sum(axis=1), np.ones(k) / k, mode="same")
    runs = _runs(profile >= max(2.0, min_ink_frac * w))

    gap = max(2, int(merge_gap_frac * h))
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    min_h = max(6, int(min_height_frac * h))
    crops = []
    for top, bottom in (r for r in merged if r[1] - r[0] >= min_h):
        pad = max(4, int(0.3 * (bottom - top)))
        t, b = max(0, top - pad), min(h, bottom + pad)
        cols = np.where(mask[top:bottom].any(axis=0))[0]
        left, right = max(0, cols.min() - pad), min(w, cols.max() + pad)
        crops.append(LineCrop(t, b, img.crop((left, t, right, b))))
    return crops


class TrOCRReader:
    """Reads text lines with Microsoft's TrOCR handwriting model (needs transformers + torch)."""

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # lazy: large download

        self._proc = TrOCRProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()

    def read_line(self, line_image: Image.Image) -> tuple[str, float | None]:
        import torch

        pixels = self._proc(images=line_image.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            out = self._model.generate(pixels, max_new_tokens=64, output_scores=True, return_dict_in_generate=True)
        text = self._proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        try:  # mean probability of the generated tokens; None if unavailable
            scores = self._model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            confidence = float(torch.exp(scores[0]).mean())
        except Exception:
            confidence = None
        return text, confidence

    def read(self, image: Image.Image) -> list[OCRLine]:
        lines = []
        for crop in segment_lines(image):
            text, conf = self.read_line(crop.image)
            lines.append(OCRLine(text, conf, crop.top, crop.bottom))
        return lines


def join_lines(lines: list[OCRLine]) -> str:
    return "\n".join(l.text for l in lines if l.text)


def doubtful_line_numbers(lines: list[OCRLine]) -> list[int]:
    """1-based numbers of non-empty lines the teacher should double-check."""
    return [i for i, l in enumerate((l for l in lines if l.text), 1) if l.doubtful]
