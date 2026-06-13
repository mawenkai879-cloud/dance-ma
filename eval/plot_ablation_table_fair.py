"""Render the FAIR hierarchical ablation (all variants: 10-epoch ft, loss weight 1.5)."""
import matplotlib.pyplot as plt

ROWS = [
    # id, method, cond, loss, PFC, beat, fid, div
    ("A", "Baseline (EDGE)",       "-", "-", 1.0192, 0.2392, 341.65, 13.82),
    ("B", "+ Hier. Conditioning",  "Y", "-", 1.7362, 0.2574, 423.34, 19.04),
    ("C", "+ Hier. Loss",          "-", "Y", 1.3229, 0.2503, 427.53, 17.11),
    ("D", "+ Both (full model)",   "Y", "Y", 0.6192, 0.2642, 170.57, 4.91),
    ("GT", "Ground Truth",         "-", "-", None,   None,   0.00,  10.21),
]
COLS = ["", "Method", "H-Cond", "H-Loss",
        "PFC \u2193", "Beat Align \u2191", "FID_k \u2193", "Div_k (\u2192GT 10.21)"]

data = ROWS[:-1]
best_pfc = min(r[4] for r in data)
best_beat = max(r[5] for r in data)
best_fid = min(r[6] for r in data)
best_div = min(data, key=lambda r: abs(r[7] - 10.21))[7]

cell_text, cell_colors = [], []
for r in ROWS:
    rid, method, cond, loss, pfc, beat, fid, div = r
    cell_text.append([
        rid, method, cond, loss,
        "-" if pfc is None else f"{pfc:.3f}",
        "-" if beat is None else f"{beat:.3f}",
        f"{fid:.1f}", f"{div:.2f}",
    ])
    rc = ["white"] * len(COLS)
    if rid == "GT":
        rc = ["#eef2f7"] * len(COLS)
    else:
        if pfc == best_pfc: rc[4] = "#a5d6a7"
        if beat == best_beat: rc[5] = "#a5d6a7"
        if fid == best_fid: rc[6] = "#a5d6a7"
        if div == best_div: rc[7] = "#a5d6a7"
    cell_colors.append(rc)

fig, ax = plt.subplots(figsize=(12, 2.9))
ax.axis("off")
tbl = ax.table(cellText=cell_text, colLabels=COLS, cellColours=cell_colors,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
for j in range(len(COLS)):
    tbl[0, j].set_facecolor("#37474f")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(len(ROWS) + 1):
    tbl[i, 1].set_text_props(ha="left")
# emphasize full model row
for j in range(len(COLS)):
    tbl[4, j].set_text_props(fontweight="bold")
ax.set_title("Controlled Ablation (all variants: 10-epoch fine-tune, hier-loss weight = 1.5, same 20 slices & noise)\n"
             "green = best per metric;  full model wins PFC / Beat Align / FID_k",
             fontsize=10.5, pad=12)
fig.tight_layout()
out = "eval/ablation/ccl_visualizations/hierarchical_ablation_fair_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved:", out)
