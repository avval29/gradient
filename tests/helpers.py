"""Shared test helpers."""
import random

from PIL import Image, ImageDraw


def fake_page(centers, w=800, h=500, seed=0, ink=0, noise=0):
    """White page with scribble 'handwriting' bands centred on the given rows."""
    rnd = random.Random(seed)
    img = Image.new("L", (w, h), 245)
    d = ImageDraw.Draw(img)
    for cy in centers:
        for _ in range(120):
            x, y = rnd.randint(40, w - 60), cy + rnd.randint(-12, 12)
            d.line([(x, y), (x + rnd.randint(6, 18), y + rnd.randint(-5, 5))], fill=ink, width=3)
    return img
