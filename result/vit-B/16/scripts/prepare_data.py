"""数据审计、去重和固定划分

协议（Proposal §3）：
  * train / val  <- GenImage SD1.4 的官方 train 分区，100 个 ImageNet 类，按类平衡真实图与 SD1.4 图。
  * test         <- 官方 val 分区：SD1.4（已见）+ SD1.5、ADM、Midjourney、BigGAN（未见）；
                    每个生成器 1000 张生成图，都与同一组 1000 张真实图（SD1.4 val/nature）配对。
  * external     <- UniversalFakeDetect 测试子集（ProGAN、LDM-200），可选，单独报告。
  * 感知哈希 (pHash) 去重：val 对 train，test/external 对 train+val，并检查真实图的 ImageNet ID。
  * 审计类别 / 分辨率 / 文件格式分布（Grommelt et al. [5]），写入 audit.csv。

输出格式见 manifests/README.md

示例：
  python -m scripts.prepare_data --data_root /project/mscaispod2026/group11 \\
     --gen sd14=genimage/sd14 --gen sd15=genimage/sd15 --gen adm=genimage/adm \\
     --gen midjourney=genimage/midjourney --gen biggan=genimage/biggan \\
     --out_dir manifests
"""
import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
WNID_RE = re.compile(r"(n\d{8})")
CLS_RE = re.compile(r"^(\d{1,4})_")  # SD1.4 fakes: e.g. 862_sdv4_00127.png -> class 862


# ----------------------------------------------------------------------------- scanning
def scan(root_abs, data_root):
    """Yield (relpath, split, label) using path components train|val and ai|nature|0_real|1_fake."""
    out = []
    for dp, _, files in os.walk(root_abs, followlinks=True):
        comps = [c.lower() for c in os.path.relpath(dp, root_abs).split(os.sep)]
        split = "train" if "train" in comps else ("val" if "val" in comps else None)
        if "ai" in comps or "1_fake" in comps:
            label = 1
        elif "nature" in comps or "0_real" in comps:
            label = 0
        else:
            label = None
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                out.append((os.path.relpath(os.path.join(dp, f), data_root), split, label))
    return out


def parse_kv(items):
    d = {}
    for it in items or []:
        k, v = it.split("=", 1)
        d[k] = v
    return d


# ----------------------------------------------------------------------------- hashing
def _info(args):
    data_root, rel = args
    from PIL import Image
    import imagehash
    try:
        with Image.open(os.path.join(data_root, rel)) as im:
            fmt, (w, h) = im.format, im.size
            ph = str(imagehash.phash(im.convert("RGB")))
        return rel, fmt, w, h, ph, True
    except Exception:
        return rel, None, -1, -1, None, False


def add_info(df, data_root, workers):
    if len(df) == 0:
        return df.assign(fmt=[], width=[], height=[], phash=[])
    with Pool(workers) as pool:
        res = pool.map(_info, [(data_root, r) for r in df["relpath"]], chunksize=64)
    info = pd.DataFrame(res, columns=["relpath", "fmt", "width", "height", "phash", "ok"])
    df = df.merge(info, on="relpath", how="left")
    bad = df[~df["ok"]]
    if len(bad):
        print(f"[warn] {len(bad)} unreadable files dropped, e.g. {bad['relpath'].iloc[0]}")
    return df[df["ok"]].drop(columns="ok").reset_index(drop=True)


def to_u64(hexes):
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lut[x.view(np.uint8)].reshape(-1, 8).sum(1).reshape(x.shape)


def near_dup(query_hex, ref_hex, thr, chunk=512):
    """Bool mask: query[i] within Hamming distance <= thr of any ref hash."""
    q, r = to_u64(query_hex), to_u64(ref_hex)
    mask = np.zeros(len(q), dtype=bool)
    if len(q) == 0 or len(r) == 0:
        return mask
    for s in range(0, len(q), chunk):
        d = popcount64(q[s:s + chunk, None] ^ r[None, :])
        mask[s:s + chunk] = (d <= thr).any(1)
    return mask


def drop_within_dups(df):
    n0 = len(df)
    df = df.drop_duplicates("phash").reset_index(drop=True)
    return df, n0 - len(df)


def stem(rel):
    return os.path.splitext(os.path.basename(rel))[0]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--gen", action="append", required=True,
                    help="name=relative_dir of a GenImage generator; must include the train generator")
    ap.add_argument("--train_gen", default="sd14")
    ap.add_argument("--ext", action="append", default=[],
                    help="name=dir (with 0_real/1_fake inside) or name=fake_dir:real_dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--n_train", type=int, default=20000, help="per label (real / AI)")
    ap.add_argument("--n_val", type=int, default=2000, help="per label")
    ap.add_argument("--n_test", type=int, default=1000, help="per generator, and number of real test images")
    ap.add_argument("--n_ext", type=int, default=1000, help="per label for each external set")
    ap.add_argument("--hamming", type=int, default=4, help="pHash near-duplicate threshold")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--wnid_list", default=None,
                    help="optional file with the 1000 ImageNet wnids in class-index order")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(a.out_dir, "train.csv")):
        raise SystemExit(f"{a.out_dir}/train.csv already exists - manifests are frozen, refusing to overwrite")
    rng = random.Random(a.seed)
    gens = parse_kv(a.gen)
    exts = parse_kv(a.ext)
    assert a.train_gen in gens, f"--gen must include {a.train_gen}"
    report = {"args": vars(a)}

    # ---- 1. scan GenImage folders
    scans = {}
    for g, rel in gens.items():  # rel may list several dirs: "dirA,dirB" (e.g. train + val downloads)
        rows = [r for d in rel.split(",") for r in scan(os.path.join(a.data_root, d), a.data_root)]
        df = pd.DataFrame(rows, columns=["relpath", "split", "label"])
        df["source"] = g
        scans[g] = df
        cnt = df.groupby(["split", "label"], dropna=False).size().to_dict()
        print(f"[scan] {g}: {len(df)} images  {cnt}")
        report[f"scan_{g}"] = {str(k): int(v) for k, v in cnt.items()}

    tr = scans[a.train_gen]
    real_tr = tr[(tr.split == "train") & (tr.label == 0)].copy()
    fake_tr = tr[(tr.split == "train") & (tr.label == 1)].copy()

    # ---- 2. ImageNet class index <-> wnid (standard order = sorted wnids)
    real_tr["wnid"] = real_tr.relpath.map(lambda r: (WNID_RE.search(os.path.basename(r)) or [None])[0])
    if a.wnid_list:
        wnids = [l.split()[0] for l in open(a.wnid_list) if l.strip()]
    else:
        wnids = sorted(real_tr.wnid.dropna().unique())
    if len(wnids) != 1000:
        raise SystemExit(f"Found {len(wnids)} wnids (need 1000); pass --wnid_list LOC_synset_mapping.txt")
    real_tr["cls"] = real_tr.wnid.map({w: i for i, w in enumerate(wnids)})
    fake_tr["cls"] = fake_tr.relpath.map(lambda r: int(m.group(1)) if (m := CLS_RE.match(os.path.basename(r))) else -1)
    print(f"[class] unparsable fake class ids: {(fake_tr.cls < 0).sum()}")

    # ---- 3. choose classes and sample a class-balanced pool
    classes = sorted(rng.sample(range(1000), a.num_classes))
    per_cls = math.ceil((a.n_train + a.n_val) / a.num_classes)
    pools = {0: [], 1: []}
    short = 0
    for c in classes:
        r = real_tr[real_tr.cls == c].relpath.tolist()
        f = fake_tr[fake_tr.cls == c].relpath.tolist()
        rng.shuffle(r); rng.shuffle(f)
        k = min(per_cls + 3, len(r), len(f))  # +3 spare for de-dup losses
        short += k < per_cls
        pools[0] += [(x, c) for x in r[:k]]
        pools[1] += [(x, c) for x in f[:k]]
    if short:
        print(f"[warn] {short}/{a.num_classes} classes have fewer than {per_cls} images per label; "
              f"train set will be smaller than requested (see report).")
    pool = pd.concat([pd.DataFrame(pools[l], columns=["relpath", "cls"]).assign(label=l) for l in (0, 1)])
    pool["source"] = np.where(pool.label == 1, a.train_gen, "real")
    pool["orig_split"] = "train"

    # ---- 4. test candidates (official val partitions, all classes)
    over = lambda n: int(n * 1.1) + 10
    tv = tr[(tr.split == "val") & (tr.label == 0)].relpath.tolist()
    rng.shuffle(tv)
    test_real = pd.DataFrame({"relpath": tv[:over(a.n_test)], "label": 0, "source": "real"})
    test_fake = {}
    for g, df in scans.items():
        v = df[(df.split == "val") & (df.label == 1)].relpath.tolist()
        rng.shuffle(v)
        test_fake[g] = pd.DataFrame({"relpath": v[:over(a.n_test)], "label": 1, "source": g})
        if len(v) < a.n_test:
            print(f"[warn] {g}: only {len(v)} val fakes")

    ext_sets = {}
    for name, spec in exts.items():
        if ":" in spec:
            fdir, rdir = spec.split(":", 1)
            f = [r for r, _, _ in scan(os.path.join(a.data_root, fdir), a.data_root)]
            r = [r for r, _, _ in scan(os.path.join(a.data_root, rdir), a.data_root)]
        else:
            rows = scan(os.path.join(a.data_root, spec), a.data_root)
            f = [r for r, _, l in rows if l == 1]
            r = [r for r, _, l in rows if l == 0]
        rng.shuffle(f); rng.shuffle(r)
        ext_sets[name] = pd.concat([
            pd.DataFrame({"relpath": r[:over(a.n_ext)], "label": 0, "source": f"{name}_real"}),
            pd.DataFrame({"relpath": f[:over(a.n_ext)], "label": 1, "source": name})])
        print(f"[ext] {name}: {len(r)} real, {len(f)} fake found")

    # ---- 5. image info + pHash
    print("[hash] computing pHash / size / format ...")
    pool = add_info(pool, a.data_root, a.workers)
    test_real = add_info(test_real, a.data_root, a.workers)
    test_fake = {g: add_info(d, a.data_root, a.workers) for g, d in test_fake.items()}
    ext_sets = {n: add_info(d, a.data_root, a.workers) for n, d in ext_sets.items()}

    # ---- 6. de-dup inside the pool, then class-stratified train/val split
    dedup = {}
    pool, dedup["pool_exact"] = drop_within_dups(pool)
    val_per_cls = max(1, round(a.n_val / a.num_classes))
    train_rows, val_rows = [], []
    for c in classes:
        sub = pool[pool.cls == c]
        r = sub[sub.label == 0].sample(frac=1, random_state=a.seed + c)
        f = sub[sub.label == 1].sample(frac=1, random_state=a.seed + c)
        k = min(len(r), len(f), per_cls)
        v = min(val_per_cls, max(0, k - 1))
        val_rows += [r.iloc[:v], f.iloc[:v]]
        train_rows += [r.iloc[v:k], f.iloc[v:k]]
    train = pd.concat(train_rows).reset_index(drop=True)
    val = pd.concat(val_rows).reset_index(drop=True)
    m = near_dup(val.phash, train.phash, a.hamming)
    dedup["val_vs_train"] = int(m.sum())
    val = val[~m].reset_index(drop=True)
    # cap to requested sizes, keeping labels balanced
    def cap(df, n):
        n = min(n, (df.label == 0).sum(), (df.label == 1).sum())
        return pd.concat([df[df.label == l].sample(n=n, random_state=a.seed) for l in (0, 1)]).reset_index(drop=True)
    train, val = cap(train, a.n_train), cap(val, a.n_val)
    train["split"], val["split"] = "train", "val"

    # ---- 7. de-dup test / external against train+val
    ref = pd.concat([train.phash, val.phash])
    ref_stems = set(map(stem, pd.concat([train.relpath, val.relpath])))

    def clean_test(df, n, key):
        m1 = near_dup(df.phash, ref, a.hamming)
        m2 = df.relpath.map(stem).isin(ref_stems).values & (df.label.values == 0)
        df2, ex = drop_within_dups(df[~(m1 | m2)])
        dedup[key] = int(m1.sum() + m2.sum() + ex)
        return df2

    test_real = clean_test(test_real, a.n_test, "test_real").head(a.n_test)
    test_real["split"] = "test"
    tests = {}
    for g, d in test_fake.items():
        d = clean_test(d, a.n_test, f"test_{g}")
        d = d[~near_dup(d.phash, test_real.phash, 0)].head(a.n_test)
        d["split"] = "test"
        tests[g] = pd.concat([test_real, d]).reset_index(drop=True)
    for n, d in ext_sets.items():
        d = clean_test(d, a.n_ext, f"ext_{n}")
        d = cap(d, a.n_ext)
        d["split"] = "external"
        tests[f"ext_{n}"] = d
    report["dedup_removed"] = dedup
    print(f"[dedup] removed: {dedup}")

    # ---- 8. write manifests
    cols = ["relpath", "label", "source", "split", "cls", "fmt", "width", "height", "phash"]
    w = lambda df, name: df.reindex(columns=cols).to_csv(os.path.join(a.out_dir, name), index=False)
    w(train, "train.csv"); w(val, "val.csv")
    for k, d in tests.items():
        w(d, f"test_{k}.csv")
    with open(os.path.join(a.out_dir, "classes.json"), "w") as fh:
        json.dump({"class_idx": classes, "wnids": [wnids[c] for c in classes]}, fh)

    # ---- 9. audit: class / resolution / format per (split, source, label)
    allm = pd.concat([train, val] + [d.assign(split=f"test_{k}") for k, d in tests.items()]).reset_index(drop=True)
    allm["is512sq"] = (allm.width == 512) & (allm.height == 512)
    allm["min_side"] = np.minimum(allm.width, allm.height)
    audit = allm.groupby(["split", "source", "label"]).agg(
        n=("relpath", "size"),
        jpeg_frac=("fmt", lambda s: float((s == "JPEG").mean())),
        png_frac=("fmt", lambda s: float((s == "PNG").mean())),
        w_median=("width", "median"), h_median=("height", "median"),
        min_side_p05=("min_side", lambda s: float(np.percentile(s, 5))),
        sq512_frac=("is512sq", "mean"),
        n_classes=("cls", "nunique")).reset_index()
    audit.to_csv(os.path.join(a.out_dir, "audit.csv"), index=False)
    print(audit.to_string(index=False))

    report["sizes"] = {"train": train.label.value_counts().to_dict(), "val": val.label.value_counts().to_dict(),
                       **{k: d.label.value_counts().to_dict() for k, d in tests.items()}}
    report["sizes"] = {k: {int(l): int(n) for l, n in v.items()} for k, v in report["sizes"].items()}
    with open(os.path.join(a.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[done] manifests in {a.out_dir}: sizes {report['sizes']}")


if __name__ == "__main__":
    main()
