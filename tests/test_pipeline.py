"""端到端冒烟测试：清单 -> 特征缓存 -> 训练 -> 评测（缓存模式与逐图预测模式）

使用随机初始化的小型 CLIP（tiny-random），不需要下载权重
  python -m unittest discover -s tests
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    result = subprocess.run([sys.executable, "-m", *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{' '.join(map(str, args))}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return result.stdout


def _make_split(root: Path, name: str, n: int, rng, source="sd14"):
    rows = []
    for i in range(n):
        for label, kind, ext in ((0, "real", "jpg"), (1, source, "png")):
            rel = f"{name}/{kind}/{i}.{ext}"
            path = root / "data" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if label == 0:
                arr = rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)
            else:
                arr = np.full((40, 40, 3), rng.integers(80, 200), np.uint8) + rng.integers(0, 10, (40, 40, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)
            rows.append(dict(relpath=rel, label=label, source=kind))
    (root / "manifests").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "manifests" / f"{name}.csv", index=False)


class PipelineTest(unittest.TestCase):
    def test_corruptions_and_metrics(self):
        from common.corruptions import EVAL_PERTURBS, apply_eval_perturb
        from common.metrics import binary_metrics

        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (30, 50, 3), dtype=np.uint8))
        for name in EVAL_PERTURBS:
            self.assertEqual(apply_eval_perturb(img, name).size, img.size)
        m = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5)
        self.assertEqual(m["acc"], 1.0)
        self.assertEqual(m["auroc"], 1.0)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(0)
            _make_split(tmp, "train", 24, rng)
            _make_split(tmp, "val", 8, rng)
            _make_split(tmp, "test_sd14", 8, rng)
            _make_split(tmp, "test_adm", 8, rng, source="adm")
            paths = ["--data_root", tmp / "data", "--manifest_dir", tmp / "manifests",
                     "--cache_dir", tmp / "cache", "--checkpoint_dir", tmp / "ckpt"]
            for name in ("baseline", "augmented"):
                text = (ROOT / "clip_detector" / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
                for old, new in (("seeds: [0, 1, 2]", "seeds: [0, 1]"), ("epochs: 400", "epochs: 5"),
                                 ("learning_rates: [0.001, 0.0003, 0.0001]", "learning_rates: [0.01]"),
                                 ("weight_decays: [0.0, 0.0001, 0.01]", "weight_decays: [0.0]"),
                                 ("num_workers: auto", "num_workers: 0"), ("batch_size: 64", "batch_size: 16")):
                    self.assertIn(old, text)
                    text = text.replace(old, new)
                cfg = tmp / f"{name}.yaml"
                cfg.write_text(text, encoding="utf-8")
                _run("clip_detector.cache_features", "--config", cfg, "--model", "tiny-random", *paths)
                _run("clip_detector.train", "--config", cfg, *paths)
            self.assertTrue((tmp / "cache" / "train_clean.pt").is_file())
            self.assertTrue((tmp / "cache" / "train_aug.pt").is_file())
            self.assertTrue((tmp / "ckpt" / "clip_aug" / "seed1" / "head.pt").is_file())

            _run("scripts.evaluate", "--runs", tmp / "ckpt" / "clip_noaug", tmp / "ckpt" / "clip_aug",
                 "--feat_dir", tmp / "cache" / "test", "--out_dir", tmp / "eval", "--save_preds",
                 "--unseen", "adm")
            summary = pd.read_csv(tmp / "eval" / "summary.csv")
            self.assertEqual(set(summary.config), {"clip_noaug", "clip_aug"})
            self.assertEqual(set(summary.test), {"sd14", "adm", "unseen_macro"})
            self.assertEqual(summary.perturb.nunique(), 8)

            # 逐图预测模式（ResNet 等模型使用的入口）必须得到相同的汇总
            preds = sorted((tmp / "eval").glob("preds_*.csv.gz"))
            _run("scripts.evaluate", "--preds", *preds, "--out_dir", tmp / "eval2", "--unseen", "adm")
            again = pd.read_csv(tmp / "eval2" / "summary.csv")
            key = ["config", "test", "perturb"]
            a = summary.sort_values(key).reset_index(drop=True)
            b = again.sort_values(key).reset_index(drop=True)
            num = a.select_dtypes("number").columns
            np.testing.assert_allclose(a[num].values, b[num].values, atol=1e-6, equal_nan=True)

            out = _run("scripts.predict", "--checkpoint", tmp / "ckpt" / "clip_aug" / "seed0" / "head.pt",
                       "--model", "tiny-random", tmp / "data" / "test_adm" / "adm" / "0.png")
            self.assertIn("P(AI)=", out)


if __name__ == "__main__":
    unittest.main()
