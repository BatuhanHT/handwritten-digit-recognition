"""predict.py testleri. Çalıştırma: python -m unittest discover -s tests -v

Model dosyası ya da MNIST indirmesi gerektirmez; rastgele başlatılmış CNN kullanır.
Yalnızca RealModelTest, eğitilmiş model ve MNIST varsa çalışır (yoksa atlanır).
"""
import base64
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import MODEL_PATH, forward_steps, load_model, pct, preprocess, run  # noqa: E402
from train import CNN  # noqa: E402


def drawing(strokes, size=280, width=20, bg=255, ink=0):
    """Beyaz tuvale siyah kalemle çizilmiş bir rakam (Electron tuvali gibi)."""
    img = Image.new("L", (size, size), bg)
    d = ImageDraw.Draw(img)
    for stroke in strokes:
        d.line(stroke, fill=ink, width=width, joint="curve")
    return img


SEVEN = [[(70, 60), (210, 60), (120, 230)]]
SMALL_ONE_TOP_LEFT = [[(40, 20), (40, 90)]]


def center_of_mass(a):
    a = a.astype(np.float64)
    ys, xs = np.indices(a.shape)
    return (ys * a).sum() / a.sum(), (xs * a).sum() / a.sum()


class PreprocessTest(unittest.TestCase):
    def test_returns_normalized_1x1x28x28_tensor(self):
        pre = preprocess(drawing(SEVEN))
        x = pre["tensor"]
        self.assertEqual(tuple(x.shape), (1, 1, 28, 28))
        self.assertAlmostEqual(float(x.min()), -0.1307 / 0.3081, places=4)  # zemin 0 → normalize

    def test_light_background_is_inverted_to_white_on_black(self):
        pre = preprocess(drawing(SEVEN))
        self.assertTrue(pre["inverted"])
        img28 = pre["img28"]
        self.assertEqual(int(img28[0, 0]), 0)
        self.assertGreater(int(img28.max()), 200)

    def test_dark_background_is_not_inverted(self):
        pre = preprocess(drawing(SEVEN, bg=0, ink=255))
        self.assertFalse(pre["inverted"])
        self.assertGreater(int(pre["img28"].max()), 200)

    def test_digit_fits_in_20px_and_centered_by_mass(self):
        img28 = preprocess(drawing(SEVEN))["img28"]
        ys, xs = np.where(img28 > 0)
        self.assertLessEqual(ys.max() - ys.min() + 1, 20)
        self.assertLessEqual(xs.max() - xs.min() + 1, 20)
        cy, cx = center_of_mass(img28)
        self.assertLess(abs(cy - 14), 1.0)
        self.assertLess(abs(cx - 14), 1.0)

    def test_off_center_small_drawing_is_moved_to_center(self):
        pre = preprocess(drawing(SMALL_ONE_TOP_LEFT))
        cy, cx = center_of_mass(pre["img28"])
        self.assertLess(abs(cy - 14), 1.0)
        self.assertLess(abs(cx - 14), 1.5)

    def test_bbox_matches_drawn_stroke(self):
        x0, y0, x1, y1 = preprocess(drawing(SMALL_ONE_TOP_LEFT, width=10))["bbox"]
        self.assertTrue(30 <= x0 <= 36 and 44 <= x1 <= 50, (x0, x1))
        self.assertTrue(10 <= y0 <= 20 and 90 <= y1 <= 100, (y0, y1))

    def test_transparent_canvas_counts_as_white(self):
        rgba = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
        ImageDraw.Draw(rgba).line(SEVEN[0], fill=(0, 0, 0, 255), width=20)
        pre = preprocess(rgba)
        self.assertTrue(pre["inverted"])

    def test_empty_canvas_raises(self):
        with self.assertRaises(ValueError):
            preprocess(drawing([]))

    def test_tiny_drawing_and_single_tap_raise(self):
        # Kalın kalemle küçük çizim ya da tek tık, ölçeklenince şişer ve emin ama yanlış tahmin verir
        for strokes in ([[(130, 130), (140, 160)]], [[(140, 140), (141, 141)]]):
            with self.assertRaises(ValueError):
                preprocess(drawing(strokes))

    def test_mnist_sized_image_is_not_rejected_as_tiny(self):
        img = Image.new("L", (28, 28), 0)
        ImageDraw.Draw(img).line([(14, 4), (14, 23)], fill=255, width=2)
        self.assertEqual(tuple(preprocess(img)["tensor"].shape), (1, 1, 28, 28))

    def test_forced_invert_ignores_dark_frame_drawn_on_canvas(self):
        # Tuvalin kenarlarını boyayan bir çizim, otomatik algılamayı yanıltırdı
        frame = [[(5, 5), (275, 5), (275, 275), (5, 275), (5, 5)], [(140, 60), (140, 220)]]
        self.assertTrue(preprocess(drawing(frame, width=24), invert=True)["inverted"])


class ForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CNN().eval()

    def test_steps_match_model_forward(self):
        x = preprocess(drawing(SEVEN))["tensor"]
        _, logits, _ = forward_steps(self.model, x)
        with torch.no_grad():
            expected = self.model(x)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))

    def test_step_shapes_match_architecture(self):
        steps, _, conv1 = forward_steps(self.model, torch.zeros(1, 1, 28, 28))
        shapes = [tuple(t.shape[1:]) for _, t in steps]
        self.assertEqual(shapes, [(32, 14, 14), (64, 7, 7), (3136,), (128,), (10,)])
        self.assertEqual(tuple(conv1.shape[1:]), (32, 28, 28))


class RunTest(unittest.TestCase):
    def test_prints_four_steps_probabilities_and_prediction(self):
        torch.manual_seed(0)
        lines = []
        res = run(CNN().eval(), drawing(SEVEN), out=lines.append)
        for step in ("[1/4]", "[2/4]", "[3/4]", "[4/4]"):
            self.assertTrue(any(l.startswith(step) for l in lines), step)
        self.assertTrue(lines[-1].startswith(f"Tahmin: {res['digit']}"))
        self.assertAlmostEqual(sum(res["probs"]), 1.0, places=4)
        self.assertEqual(res["digit"], int(np.argmax(res["probs"])))
        self.assertEqual(len(res["images"]["maps"]), 8)

    def test_pct_uses_turkish_format_and_never_shows_100(self):
        self.assertEqual(pct(0.9921), "%99,2")
        self.assertEqual(pct(0.0), "%0,0")
        self.assertEqual(pct(0.99996), "%99,9+")  # %100,0 gösterilmez


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class ServeProtocolTest(unittest.TestCase):
    """Electron'un kullandığı --serve kipi: READY, id'nin geri dönmesi, hatadan sonra ayakta kalma."""

    def test_ready_results_and_errors_keep_process_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(0)
            model_file = Path(tmp) / "model.pt"
            torch.save(CNN().state_dict(), model_file)
            requests = [
                json.dumps({"id": 1, "png": png_b64(drawing(SEVEN))}),
                "bu json değil",
                json.dumps({"id": 3, "png": png_b64(drawing([]))}),
                json.dumps({"id": 4, "png": png_b64(drawing(SEVEN))}),
            ]
            proc = subprocess.run(
                [sys.executable, "-u", str(ROOT / "src" / "predict.py"), "--serve", "--model", str(model_file)],
                input="\n".join(requests) + "\n", capture_output=True, text=True, timeout=120, cwd=ROOT,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        tags = [line.split(" ", 1) for line in proc.stdout.splitlines() if line.startswith("@@")]
        kinds = [(t[0], json.loads(t[1]).get("id")) for t in tags]
        self.assertEqual(kinds, [("@@READY", None), ("@@RESULT", 1), ("@@ERROR", None), ("@@ERROR", 3), ("@@RESULT", 4)])
        self.assertIn("Tahmin:", proc.stdout)

    def test_missing_model_reports_fatal(self):
        proc = subprocess.run(
            [sys.executable, "-u", str(ROOT / "src" / "predict.py"), "--serve", "--model", "/yok/model.pt"],
            input="", capture_output=True, text=True, timeout=60, cwd=ROOT,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("@@FATAL", proc.stdout)


MNIST_RAW = ROOT / "data" / "MNIST" / "raw"


@unittest.skipUnless(MODEL_PATH.exists() and MNIST_RAW.exists(), "eğitilmiş model ya da MNIST yok")
class RealModelTest(unittest.TestCase):
    """Ön işleme gerçek modele uygun girdi üretiyor mu: MNIST rakamları tuvale büyütülüp geri tanınmalı."""

    def test_upscaled_mnist_digits_are_recognized(self):
        from torchvision import datasets

        ds = datasets.MNIST(ROOT / "data", train=False, download=False)
        model = load_model()
        n, correct = 300, 0
        for i in range(n):
            digit, label = ds[i]
            canvas = Image.eval(digit.resize((280, 280), Image.Resampling.BILINEAR), lambda v: 255 - v)
            x = preprocess(canvas, invert=True)["tensor"]
            with torch.no_grad():
                correct += int(model(x).argmax(1).item() == label)
        self.assertGreaterEqual(correct / n, 0.95)


if __name__ == "__main__":
    unittest.main()
