"""Render the hierarchical ablation results as a paper-style table figure."""
import matplotlib.pyplot as plt

ROWS = [
    # id, method, cond, loss, w, ep, PFC, beat, fid, div
    ("A",  "Baseline (EDGE)",        "-", "-", "-",   "5",  1.2785, 0.2214, 472.48, 15.40),
    ("B",  "+ Hier. Conditioning",   "Y", "-", "-",   "5",  0.8484, 0.2623, 161.94, 13.33),
    ("C",  "+ Hier. Loss",           "-", "Y", "3.0", "5",  1.0365, 0.2603, 410.09, 15.51),
    ("D",  "+ Both",                 "Y", "Y", "3.0", "5",  1.6345, 0.3060, 2282.59, 33.01),
    ("D2", "+ Both (tuned)",         "Y", "Y", "1.5", "10", 0.6192, 0.2642, 170.57, 4.91),
    ("D2", "+ Both (tuned)",         "Y", "Y", "1.5", "25", 1.1297, 0.2258, 176.48, 9.59),
    ("GT", "Ground Truth",           "-", "-", "-",   "-",  None,   None,   0.00,  10.21),
]
COLS = ["", "Method", "H-Cond", "H-Loss", "w", "Ep",
        "PFC \u2193", "Beat Align \u2191", "FID_k \u2193", "Div_k (GT 10.21)"]

# best per metric (excluding GT): PFC min, beat max, fid min, div closest to 10.21
data_rows = ROWS[:-1]
best_pfc = min(r[6] for r in data_rows)
best_beat = max(r[7] for r in data_rows)
best_fid = min(r[8] for r in data_rows)
best_div = min(data_rows, key=lambda r: abs(r[9] - 10.21))[9]

cell_text, cell_colors = [], []
for r in ROWS:
    rid, method, cond, loss, w, ep, pfc, beat, fid, div = r
    pfc_s = "-" if pfc is None else f"{pfc:.3f}"
    beat_s = "-" if beat is None else f"{beat:.3f}"
    fid_s = f"{fid:.1f}"
    div_s = f"{div:.2f}"
    cell_text.append([rid, method, cond, loss, w, ep, pfc_s, beat_s, fid_s, div_s])
    rc = ["white"] * len(COLS)
    if rid == "GT":
        rc = ["#eef2f7"] * len(COLS)
    else:
        if pfc == best_pfc: rc[6] = "#c8e6c9"
        if beat == best_beat: rc[7] = "#c8e6c9"
        if fid == best_fid: rc[8] = "#c8e6c9"
        if div == best_div: rc[9] = "#c8e6c9"
    cell_colors.append(rc)

fig, ax = plt.subplots(figsize=(13, 3.4))
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=COLS, cellColours=cell_colors,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)
for j in range(len(COLS)):
    tbl[0, j].set_facecolor("#37474f")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(len(ROWS) + 1):
    tbl[i, 1].set_text_props(ha="left")
ax.set_title("Hierarchical Ablation on AIST++ test (20 slices, same noise, 5-ep ft unless noted)\n"
             "green = best per metric;  H-Cond/H-Loss = our two proposed modules",
             fontsize=11, pad=14)
fig.tight_layout()
out = "eval/ablation/ccl_visualizations/hierarchical_ablation_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved:", out)
