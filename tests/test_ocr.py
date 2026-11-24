import random

import pytest
from PIL import Image

from gradient.evaluate import edit_distance, ocr_error_rates
from tests.helpers import fake_page
from gradient.ocr import OCRLine, doubtful_line_numbers, join_lines, otsu_threshold, segment_lines


def test_three_text_lines_are_found_in_order():
    lines = segment_lines(fake_page([100, 230, 360]))
    assert len(lines) == 3
    centers = [(l.top + l.bottom) / 2 for l in lines]
    assert centers == sorted(centers)
    for c, expected in zip(centers, [100, 230, 360]):
        assert abs(c - expected) < 30


def test_blank_page_has_no_lines():
    assert segment_lines(Image.new("L", (800, 500), 240)) == []


def test_faint_noise_is_ignored():
    rnd = random.Random(1)
    img = Image.new("L", (800, 500), 240)
    for _ in range(3000):
        img.putpixel((rnd.randint(0, 799), rnd.randint(0, 499)), 240 - rnd.randint(0, 8))
    assert segment_lines(img) == []


def test_uneven_lighting_does_not_hide_the_text():
    page = fake_page([150, 330])
    shade = Image.linear_gradient("L").resize(page.size).point(lambda v: 90 + v // 2)  # dark -> light
    shaded = Image.composite(page, shade, Image.new("L", page.size, 255)).point(lambda v: v)
    for x in range(0, 800, 2):  # darken the page progressively to mimic a shadow
        for y in range(0, 500, 5):
            shaded.putpixel((x, y), max(0, shaded.getpixel((x, y)) - x // 12))
    assert len(segment_lines(shaded)) == 2


def test_crops_cover_their_band_with_some_padding():
    (line,) = segment_lines(fake_page([250]))
    assert line.top < 250 < line.bottom and line.image.height == line.bottom - line.top


def test_otsu_separates_two_clusters():
    import numpy as np

    vals = np.concatenate([np.full(500, 10.0), np.full(500, 200.0)])
    assert 10 < otsu_threshold(vals) < 200


def test_edit_distance_known_cases():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1 and edit_distance("", "abc") == 3


def test_error_rates():
    r = ocr_error_rates(["plants make food"], ["plants make food"])
    assert r["cer"] == 0 and r["wer"] == 0
    r = ocr_error_rates(["plnts make food"], ["plants make food"])
    assert r["cer"] == pytest.approx(100 / 16) and r["wer"] == pytest.approx(100 / 3)
    with pytest.raises(ValueError):
        ocr_error_rates(["a"], [""])


def test_join_and_doubtful_lines():
    lines = [OCRLine("first", 0.9, 0, 10), OCRLine("", 0.1, 10, 20), OCRLine("second", 0.3, 20, 30),
             OCRLine("third", None, 30, 40)]
    assert join_lines(lines) == "first\nsecond\nthird"
    assert doubtful_line_numbers(lines) == [2]  # numbering skips the empty line; None confidence is not flagged
