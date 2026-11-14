"""Turn results/summary.csv into report-ready figures (PDF+PNG) and LaTeX tables.

  python -m scripts.make_report_assets --summary outputs/eval/summary.csv --out results/clip

Outputs (CVPR-friendly: vector PDF for LaTeX, PNG for preview):
  fig_cross_generator.{pdf,png}  balanced accuracy & AUROC per test generator, w/o vs with aug
  fig_robustness.{pdf,png}       balanced accuracy vs perturbation strength (seen vs unseen-avg)
  tab_clean.tex                  clean cross-generator table (mean over seeds, max std in caption, in %)
  (tables need \\usepackage{booktabs,graphicx}; each fits one CVPR column)
  tab_robustness.tex             robustness table (seen SD1.4 / unseen average)
  tab_errors.tex                 real-image accuracy vs AI recall on clean images
  key_numbers.md                 the headline numbers quoted in the text
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Validated categorical pair (blue / orange): CVD dE 24.7, normal-vision dE 33.6.
C = {"clip_noaug": "#2a78d6", "clip_aug": "#eb6834"}
NAME = {"clip_noaug": "CLIP w/o aug", "clip_aug": "CLIP + aug"}
TESTS = [("sd14", "SD1.4$^\\dagger$"), ("sd15", "SD1.5"), ("midjourney", "Midjourney"),
         ("adm", "ADM"), ("biggan", "BigGAN"), ("unseen_macro", "Unseen avg.")]
TESTS_PLAIN = {"sd14": "SD1.4 (seen)", "sd15": "SD1.5", "midjourney": "Midjourney",
               "adm": "ADM", "biggan": "BigGAN", "unseen_macro": "Unseen avg."}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def get(df, cfg, test, pert, col):
    r = df[(df.config == cfg) & (df.test == test) & (df.perturb == pert)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def pm(m, s, pct=True):
    k = 100 if pct else 1
    return f"{m * k:.1f}\\,$\\pm$\\,{s * k:.1f}" if pct else f"{m:.3f}"


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- figures
def fig_cross_generator(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), sharey=False)
    labels = [TESTS_PLAIN[t] for t, _ in TESTS]
    x = range(len(TESTS))
    w = 0.36
    for ax, (col, title) in zip(axes, [("bal_acc", "Balanced accuracy"), ("auroc", "AUROC")]):
        for i, cfg in enumerate(["clip_noaug", "clip_aug"]):
            m = [get(df, cfg, t, "clean", f"{col}_mean") for t, _ in TESTS]
            s = [get(df, cfg, t, "clean", f"{col}_std") for t, _ in TESTS]
            pos = [xi + (i - 0.5) * (w + 0.03) for xi in x]
            ax.bar(pos, m, width=w, color=C[cfg], label=NAME[cfg], zorder=2,
                   yerr=s, error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=1.5))
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)  # chance level (stated in caption)
        ax.axvline(0.5, color=GRID, lw=0.8, zorder=0)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.97))
    save(fig, out, "fig_cross_generator")


def fig_robustness(df, out):
    fams = [("JPEG quality", ["clean", "jpeg90", "jpeg70", "jpeg50"], ["none", "90", "70", "50"]),
            ("Downsample + restore", ["clean", "down0.5", "down0.25"], ["1×", "0.5×", "0.25×"]),
            ("Gaussian blur σ", ["clean", "blur0.5", "blur1.0"], ["0", "0.5", "1.0"])]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), sharey=True)
    for ax, (title, perts, ticks) in zip(axes, fams):
        for cfg in ["clip_noaug", "clip_aug"]:
            for test, ls, mk in [("sd14", "-", "o"), ("unseen_macro", (0, (4, 2)), "s")]:
                y = [get(df, cfg, test, p, "bal_acc_mean") for p in perts]
                ax.plot(range(len(perts)), y, color=C[cfg], ls=ls, lw=2, marker=mk, ms=4.5,
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xticks(range(len(perts)))
        ax.set_xticklabels(ticks)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("Balanced accuracy")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C["clip_noaug"], lw=2, label=NAME["clip_noaug"]),
               Line2D([], [], color=C["clip_aug"], lw=2, label=NAME["clip_aug"]),
               Line2D([], [], color=MUTED, lw=2, marker="o", ms=4, label="SD1.4 (seen)"),
               Line2D([], [], color=MUTED, lw=2, ls=(0, (4, 2)), marker="s", ms=4, label="Unseen avg.")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_robustness")


# ----------------------------------------------------------------------------- tables
def tab_clean(df, out):
    cl = df[(df.perturb == "clean") & df.config.isin(["clip_noaug", "clip_aug"])]
    maxstd = 100 * cl[["bal_acc_std", "auroc_std", "ece_std"]].max().max()
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{CLIP w/o aug} & \multicolumn{3}{c}{CLIP + aug} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Test set & Bal.\ Acc & AUROC & ECE$\downarrow$ & Bal.\ Acc & AUROC & ECE$\downarrow$ \\", r"\midrule"]
    for t, lab in TESTS:
        if t == "unseen_macro":
            L.append(r"\midrule")
        cells = []
        for cfg in ["clip_noaug", "clip_aug"]:
            for col in ["bal_acc", "auroc", "ece"]:
                cells.append(f"{get(df, cfg, t, 'clean', f'{col}_mean') * 100:.1f}")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Clean cross-generator results (\%, mean over 3 seeds; std $\le$" + f"{maxstd:.1f}" + r" pt). Trained on SD1.4 only "
          r"($^\dagger$seen generator); each test set pairs 1{,}000 generated images with the same 1{,}000 "
          r"held-out real ImageNet images. ECE after validation-only temperature scaling.}",
          r"\label{tab:clean}", r"\end{table}"]
    open(os.path.join(out, "tab_clean.tex"), "w").write("\n".join(L) + "\n")


def tab_robustness(df, out):
    perts = [("clean", "Clean"), ("jpeg90", "JPEG $q$=90"), ("jpeg70", "JPEG $q$=70"), ("jpeg50", "JPEG $q$=50"),
             ("down0.5", r"Down $0.5\times$"), ("down0.25", r"Down $0.25\times$"),
             ("blur0.5", r"Blur $\sigma$=0.5"), ("blur1.0", r"Blur $\sigma$=1.0")]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{SD1.4 Bal.\ Acc} & \multicolumn{2}{c}{Unseen Bal.\ Acc} & \multicolumn{2}{c}{Unseen AUROC} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Perturbation & w/o & +aug & w/o & +aug & w/o & +aug \\", r"\midrule"]
    for p, lab in perts:
        row = []
        for t, col in [("sd14", "bal_acc"), ("unseen_macro", "bal_acc"), ("unseen_macro", "auroc")]:
            a = get(df, "clip_noaug", t, p, f"{col}_mean")
            b = get(df, "clip_aug", t, p, f"{col}_mean")
            fa, fb = f"{a * 100:.1f}", f"{b * 100:.1f}"
            if b > a + 0.005:
                fb = r"\textbf{" + fb + "}"
            elif a > b + 0.005:
                fa = r"\textbf{" + fa + "}"
            row += [fa, fb]
        L.append(f"{lab} & " + " & ".join(row) + r" \\")
        if p in ("clean", "jpeg50", "down0.25"):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Robustness to test-time perturbations (\%, mean over 3 seeds; bold = better by $>$0.5 pt). "
          r"Unseen = average over SD1.5, Midjourney, ADM and BigGAN. Note: the training augmentation ranges "
          r"(JPEG $q\in[50,100]$, scale $\in[0.25,1]$, $\sigma\in[0.1,1.5]$) cover these test perturbations.}",
          r"\label{tab:robust}", r"\end{table}"]
    open(os.path.join(out, "tab_robustness.tex"), "w").write("\n".join(L) + "\n")


def real_acc(df, cfg, t, p="clean"):
    # balanced accuracy = (real_acc + ai_recall) / 2
    return 2 * get(df, cfg, t, p, "bal_acc_mean") - get(df, cfg, t, p, "recall_mean")


def tab_errors(df, out):
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{CLIP w/o aug} & \multicolumn{2}{c}{CLIP + aug} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Test set & Real acc. & AI recall & Real acc. & AI recall \\", r"\midrule"]
    for t, lab in TESTS[:-1]:
        cells = []
        for cfg in ["clip_noaug", "clip_aug"]:
            cells += [f"{real_acc(df, cfg, t) * 100:.1f}", f"{get(df, cfg, t, 'clean', 'recall_mean') * 100:.1f}"]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\caption{Per-class accuracy on clean images (\%). Real images are identical across test sets, so "
          r"cross-generator errors are almost entirely missed generated images.}",
          r"\label{tab:errors}", r"\end{table}"]
    open(os.path.join(out, "tab_errors.tex"), "w").write("\n".join(L) + "\n")


def key_numbers(df, out):
    g = lambda c, t, p, k: get(df, c, t, p, k) * 100
    lines = ["# Key numbers (balanced accuracy %, mean over 3 seeds)", ""]
    for cfg in ["clip_noaug", "clip_aug"]:
        lines.append(f"## {NAME[cfg]}")
        for t, _ in TESTS:
            lines.append(f"- {TESTS_PLAIN[t]}: bal.acc {g(cfg, t, 'clean', 'bal_acc_mean'):.1f}, "
                         f"AUROC {g(cfg, t, 'clean', 'auroc_mean'):.1f}, ECE {g(cfg, t, 'clean', 'ece_mean'):.1f}, "
                         f"real acc {real_acc(df, cfg, t) * 100:.1f}, AI recall {g(cfg, t, 'clean', 'recall_mean'):.1f}")
        lines.append("")
    lines += ["## JPEG q=50 drop (clean -> jpeg50, bal.acc points)"]
    for cfg in ["clip_noaug", "clip_aug"]:
        for t in ["sd14", "unseen_macro"]:
            lines.append(f"- {NAME[cfg]}, {TESTS_PLAIN[t]}: {g(cfg, t, 'clean', 'bal_acc_mean'):.1f} -> "
                         f"{g(cfg, t, 'jpeg50', 'bal_acc_mean'):.1f} "
                         f"({g(cfg, t, 'jpeg50', 'bal_acc_mean') - g(cfg, t, 'clean', 'bal_acc_mean'):+.1f})")
    open(os.path.join(out, "key_numbers.md"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="report_assets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.summary)
    fig_cross_generator(df, a.out)
    fig_robustness(df, a.out)
    tab_clean(df, a.out)
    tab_robustness(df, a.out)
    tab_errors(df, a.out)
    key_numbers(df, a.out)
    print("wrote:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
