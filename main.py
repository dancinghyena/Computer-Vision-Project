import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional
import bisect
from collections import Counter

@dataclass
class CoinClassification:

    cx: float
    cy: float
    radius: float

    size_px: int                                          
    size_label: str                                                   
    size_percentile: float                                                           

    mean_hue: float                                         
    mean_sat: float                            
    mean_val: float                                         
    mean_lab_a: float                                                      
    mean_lab_b: float                                                        
    color_label: str                                                              

    annotation_color: tuple = field(default_factory=lambda: (0, 255, 0))

def preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (15, 15), 3)
    return gray, blurred

def white_balance(image_bgr):

    img = image_bgr.astype(np.float32)
    mean_b = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_r = img[:, :, 2].mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    if mean_b > 1e-6: img[:, :, 0] *= mean_gray / mean_b
    if mean_g > 1e-6: img[:, :, 1] *= mean_gray / mean_g
    if mean_r > 1e-6: img[:, :, 2] *= mean_gray / mean_r
    return np.clip(img, 0, 255).astype(np.uint8)

def detect_circles(blurred, dp=1.2, min_dist_factor=0.15,
                   param1=80, param2=35, min_r_factor=0.05, max_r_factor=0.35):
    h, w = blurred.shape
    min_dist = int(w * min_dist_factor)
    min_r    = int(w * min_r_factor)
    max_r    = int(w * max_r_factor)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r,
    )
    return circles

def filter_circles(circles_raw, gray_image, edge_threshold=30.0,
                   contrast_threshold=2.0, coherence_threshold=0.65):
    if circles_raw is None:
        return None
    grad_x   = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y   = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    h, w = gray_image.shape
    kept = []
    yy, xx = np.ogrid[:h, :w]

    for cx, cy, r in circles_raw[0]:
        cx, cy, r = float(cx), float(cy), float(r)
        ri = int(round(r))
        if ri < 5:
            continue
        n_samples = max(36, int(2 * np.pi * ri * 0.3))
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys, angles_v = xs[valid], ys[valid], angles[valid]
        if len(xs) < n_samples * 0.5:
            continue
        mean_edge = grad_mag[ys, xs].mean()
        if mean_edge < edge_threshold:
            continue
        gx_vals, gy_vals = grad_x[ys, xs], grad_y[ys, xs]
        rx, ry = np.cos(angles_v), np.sin(angles_v)
        dot    = gx_vals * rx + gy_vals * ry
        g_mags = grad_mag[ys, xs]
        nonzero = g_mags > 1.0
        if nonzero.sum() < n_samples * 0.3:
            continue
        cos_angle = np.abs(dot[nonzero]) / g_mags[nonzero]
        coherence = (cos_angle > 0.707).sum() / float(nonzero.sum())
        if coherence < coherence_threshold:
            continue
        dist_sq     = (xx - cx) ** 2 + (yy - cy) ** 2
        inner_mask  = dist_sq <= (0.85 * r) ** 2
        outer_mask  = (dist_sq > (1.1 * r) ** 2) & (dist_sq <= (1.4 * r) ** 2)
        inner_pixels = gray_image[inner_mask]
        outer_pixels = gray_image[outer_mask]
        if inner_pixels.size < 10 or outer_pixels.size < 10:
            continue
        contrast = abs(float(inner_pixels.mean()) - float(outer_pixels.mean()))
        if contrast < contrast_threshold:
            continue
        kept.append([cx, cy, r, mean_edge])

    if not kept:
        return None

    kept.sort(key=lambda c: c[3], reverse=True)
    final = []
    for cand in kept:
        cx1, cy1, r1, _ = cand
        suppress = False
        for accepted in final:
            cx2, cy2, r2, _ = accepted
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if dist < max(r1, r2) * 0.7:
                suppress = True
                break
        if not suppress:
            final.append(cand)

    if not final:
        return None

    final3 = [[c[0], c[1], c[2]] for c in final]
    return np.array([[final3]], dtype=np.float32).reshape(1, -1, 3)

_COLOR_RULES = [

    ("copper",   5,     20,      80,     60,       10),                  
    ("gold",    18,     38,      60,     80,        8),                
    ("silver",   0,    180,       0,    120,       -5),                           
]

def _sample_coin_region(image_bgr, cx, cy, r, inner_fraction=0.75):

    h_img, w_img = image_bgr.shape[:2]
    yy, xx = np.ogrid[:h_img, :w_img]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * inner_fraction) ** 2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    h_vals = hsv[:, :, 0][mask]
    s_vals = hsv[:, :, 1][mask]
    v_vals = hsv[:, :, 2][mask]
    a_vals = lab[:, :, 1][mask]                        
    b_vals = lab[:, :, 2][mask]                          

    if len(h_vals) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    h_rad   = np.deg2rad(h_vals * 2.0)                                            
    mean_sin = np.sin(h_rad).mean()
    mean_cos = np.cos(h_rad).mean()
    circ_hue = (np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360) / 2.0                   

    return (
        float(circ_hue),
        float(s_vals.mean()),
        float(v_vals.mean()),
        float(a_vals.mean() - 128),                    
        float(b_vals.mean() - 128),
    )

def classify_color(mean_hue, mean_sat, mean_val, mean_lab_a, mean_lab_b):

    if 5 <= mean_hue <= 22 and mean_sat >= 60 and mean_lab_b >= 5:
        return "copper", (30, 90, 185)                              

    if 18 <= mean_hue <= 42 and mean_sat >= 50 and mean_val >= 70 and mean_lab_b >= 8:
        return "gold", (0, 200, 220)                                 

    if mean_sat < 55 or mean_val >= 170:
        return "silver", (200, 200, 200)                            

    if mean_lab_b > 5:
        return "gold", (0, 200, 220)
    return "silver", (200, 200, 200)

def classify_size(radius_px, all_radii):

    if len(all_radii) == 0:
        return "medium", 0.5

    sorted_r = sorted(all_radii)
    n = len(sorted_r)

    rank = bisect.bisect_left(sorted_r, radius_px)
    percentile = rank / max(n - 1, 1)

    if n == 1:
        return "medium", 0.5
    elif percentile < 0.34:
        return "small", percentile
    elif percentile < 0.67:
        return "medium", percentile
    else:
        return "large", percentile

def classify_coins(circles_filtered, image_bgr):

    if circles_filtered is None or len(circles_filtered[0]) == 0:
        return []

    raw_list = circles_filtered[0]                                 
    all_radii = [float(c[2]) for c in raw_list]

    results = []
    for cx, cy, r in raw_list:
        cx, cy, r = float(cx), float(cy), float(r)

        mh, ms, mv, ma, mb = _sample_coin_region(image_bgr, cx, cy, r)
        color_label, ann_color = classify_color(mh, ms, mv, ma, mb)

        size_label, size_pct = classify_size(r, all_radii)

        results.append(CoinClassification(
            cx=cx, cy=cy, radius=r,
            size_px=int(round(r)),
            size_label=size_label,
            size_percentile=size_pct,
            mean_hue=mh, mean_sat=ms, mean_val=mv,
            mean_lab_a=ma, mean_lab_b=mb,
            color_label=color_label,
            annotation_color=ann_color,
        ))

    return results

_CLASS_BGR = {
    "gold":    (0,   200, 220),
    "silver":  (200, 200, 200),
    "copper":  (30,  90,  185),
    "unknown": (0,   255,   0),
}

_SIZE_SYMBOLS = {"small": "Sml", "medium": "Med", "large": "Lrg"}

def draw_results(image_bgr, coin_classifications):

    output = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, coin in enumerate(coin_classifications, 1):
        cx, cy, r = int(coin.cx), int(coin.cy), int(coin.radius)
        color = coin.annotation_color

        cv2.circle(output, (cx, cy), r, color, 3)
        cv2.circle(output, (cx, cy), 4, color, -1)

        color_name = coin.color_label.capitalize()
        size_sym   = _SIZE_SYMBOLS.get(coin.size_label, "???")
        line1 = f"#{i}"
        line2 = f"{color_name} ({size_sym})"

        font_scale = max(0.45, r / 65)
        thickness  = 2
        (tw1, th1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(line2, font, font_scale * 0.85, thickness - 1)

        pad   = 5
        box_w = max(tw1, tw2) + pad * 2
        box_h = th1 + th2 + pad * 3
        tx    = cx - box_w // 2
        ty    = cy - r - box_h - 6                                    

        tx = max(0, min(tx, image_bgr.shape[1] - box_w))
        ty = max(0, ty)

        cv2.rectangle(output, (tx, ty), (tx + box_w, ty + box_h), (0, 0, 0), -1)
        cv2.putText(output, line1,
                    (tx + pad, ty + pad + th1),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(output, line2,
                    (tx + pad, ty + pad * 2 + th1 + th2),
                    font, font_scale * 0.85, color, thickness - 1, cv2.LINE_AA)

    return output

def build_report_figure(image_bgr, blurred, circles_raw, annotated,
                        coin_classifications, output_path):

    n = len(coin_classifications)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    title_kw = dict(color="white", fontsize=11, fontweight="bold", pad=8)

    axes[0, 0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1 · Original Image", **title_kw)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(blurred, cmap="gray")
    axes[0, 1].set_title("2 · Blurred (CLAHE + Gaussian)", **title_kw)
    axes[0, 1].axis("off")

    hough_vis = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    if circles_raw is not None:
        for x, y, r in np.uint16(np.around(circles_raw[0])):
            cv2.circle(hough_vis, (x, y), r, (0, 220, 120), 2)
            cv2.circle(hough_vis, (x, y), 3, (255, 80, 80), -1)
    axes[0, 2].imshow(hough_vis)
    axes[0, 2].set_title(f"3 · Hough Circles (raw candidates)", **title_kw)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"4 · Final: {n} Coins Detected", **title_kw)
    axes[1, 0].axis("off")

    ax5 = axes[1, 1]
    ax5.set_facecolor("#1a1a2e")
    ax5.set_title("5 · Color Features (Hue vs Saturation)", **title_kw)
    ax5.set_xlabel("Mean Hue (°×2)", color="white", fontsize=9)
    ax5.set_ylabel("Mean Saturation", color="white", fontsize=9)
    ax5.tick_params(colors="white")

    ax5.axvspan(5,  22, alpha=0.15, color="#B94020", label="_copper zone")
    ax5.axvspan(18, 42, alpha=0.12, color="#D4A017", label="_gold zone")

    _scatter_rgb = {"gold": "#E8C020", "silver": "#C0C0C0",
                    "copper": "#B87333", "unknown": "#00FF00"}
    for i, coin in enumerate(coin_classifications, 1):
        sc = _scatter_rgb.get(coin.color_label, "#00FF00")
        ax5.scatter(coin.mean_hue * 2, coin.mean_sat,
                    color=sc, s=110, zorder=5,
                    edgecolors="white", linewidths=0.8)
        ax5.annotate(str(i), (coin.mean_hue * 2, coin.mean_sat),
                     fontsize=7, color="white",
                     textcoords="offset points", xytext=(4, 4))

    legend_elements = [
        mpatches.Patch(facecolor="#E8C020", label="Gold"),
        mpatches.Patch(facecolor="#C0C0C0", label="Silver"),
        mpatches.Patch(facecolor="#B87333", label="Copper"),
    ]
    ax5.legend(handles=legend_elements, facecolor="#0f0f1a",
               labelcolor="white", fontsize=8, loc="upper right")
    ax5.set_xlim(0, 360)
    ax5.set_ylim(0, 260)
    ax5.grid(True, color="#333", linewidth=0.5)

    ax6 = axes[1, 2]
    ax6.set_facecolor("#1a1a2e")
    ax6.set_title("6 · Size Distribution (radius px)", **title_kw)
    ax6.tick_params(colors="white")
    ax6.set_xlabel("Coin #", color="white", fontsize=9)
    ax6.set_ylabel("Radius (px)", color="white", fontsize=9)
    ax6.grid(axis="y", color="#333", linewidth=0.5)

    if coin_classifications:
        indices = list(range(1, n + 1))
        radii   = [c.radius for c in coin_classifications]
        size_colors = {"small": "#6699CC", "medium": "#55AA88", "large": "#CC7744"}
        bar_colors  = [size_colors.get(c.size_label, "#888888") for c in coin_classifications]
        bars = ax6.bar(indices, radii, color=bar_colors, edgecolor="#444", linewidth=0.8)
        for bar, coin in zip(bars, coin_classifications):
            ax6.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     coin.size_label[0].upper(),
                     ha="center", va="bottom", color="white", fontsize=8)

        size_patches = [
            mpatches.Patch(facecolor="#6699CC", label="Small"),
            mpatches.Patch(facecolor="#55AA88", label="Medium"),
            mpatches.Patch(facecolor="#CC7744", label="Large"),
        ]
        ax6.legend(handles=size_patches, facecolor="#0f0f1a",
                   labelcolor="white", fontsize=8, loc="upper right")
        ax6.set_xticks(indices)
        ax6.xaxis.label.set_color("white")

    fig.suptitle(f"Coin Counting & Classification — Total: {n}",
                 color="white", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Report saved -> {output_path}")

def _load_gt_circles(label_path, img_w, img_h):

    if not os.path.isfile(label_path):
        return []
    circles = []
    with open(label_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx_n, cy_n, w_n, h_n = map(float, parts)
            circles.append((
                cx_n * img_w,
                cy_n * img_h,
                min(w_n * img_w, h_n * img_h) / 2.0,
            ))
    return circles

def _circle_iou(cx1, cy1, r1, cx2, cy2, r2):

    ix1 = max(cx1 - r1, cx2 - r2);  iy1 = max(cy1 - r1, cy2 - r2)
    ix2 = min(cx1 + r1, cx2 + r2);  iy2 = min(cy1 + r1, cy2 + r2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (2 * r1) ** 2 + (2 * r2) ** 2 - inter
    return inter / union if union > 0 else 0.0

def evaluate_detection(coin_classifications, label_path, img_w, img_h,
                       iou_thresh=0.4):

    gt  = _load_gt_circles(label_path, img_w, img_h)
    det = [(c.cx, c.cy, c.radius) for c in coin_classifications]

    matched_gt = set()
    matched_det = set()
    for di, (dx, dy, dr) in enumerate(det):
        best_iou, best_gi = 0.0, -1
        for gi, (gx, gy, gr) in enumerate(gt):
            if gi in matched_gt:
                continue
            iou = _circle_iou(dx, dy, dr, gx, gy, gr)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh:
            matched_gt.add(best_gi)
            matched_det.add(di)

    tp   = len(matched_gt)
    fp   = len(det) - tp
    fn   = len(gt)  - tp
    prec = tp / len(det) if det else 0.0
    rec  = tp / len(gt)  if gt  else 0.0
    return dict(gt_count=len(gt), det_count=len(det),
                tp=tp, fp=fp, fn=fn, precision=prec, recall=rec)

def run_pipeline(image_path, output_dir="output",
                 dp=1.2, param1=80, param2=35,
                 min_dist_factor=0.15, min_r_factor=0.05, max_r_factor=0.25,
                 edge_threshold=30.0, contrast_threshold=2.0,
                 coherence_threshold=0.65, labels_dir=None):

    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    print(f"[OK] Loaded {image_path}  ({image_bgr.shape[1]}x{image_bgr.shape[0]})")

    image_bgr = white_balance(image_bgr)

    gray, blurred = preprocess(image_bgr)

    circles_raw = detect_circles(blurred, dp=dp,
                                 min_dist_factor=min_dist_factor,
                                 param1=param1, param2=param2,
                                 min_r_factor=min_r_factor,
                                 max_r_factor=max_r_factor)

    circles_filtered = filter_circles(circles_raw, gray,
                                      edge_threshold=edge_threshold,
                                      contrast_threshold=contrast_threshold,
                                      coherence_threshold=coherence_threshold)

    n_before = 0 if circles_raw is None else len(circles_raw[0])
    if circles_filtered is None:
        if n_before > 0:
            print(f"[!] Hough found {n_before} candidate(s) but all rejected by filter.")
        else:
            print("[!] No circles detected.")
        coin_classifications = []
    else:
        n_after = len(circles_filtered[0])
        print(f"[OK] Hough candidates: {n_before} -> after filter: {n_after}")

        coin_classifications = classify_coins(circles_filtered, image_bgr)

        print("\n+-------------------------------------------------------------+")
        print("|  #   cx     cy    r(px)   color     size    H    Sat   Val  |")
        print("+-------------------------------------------------------------+")
        for i, c in enumerate(coin_classifications, 1):
            print(f"| {i:2d}  {int(c.cx):4d}  {int(c.cy):4d}   {c.size_px:4d}   "
                  f"{c.color_label:<8s}  {c.size_label:<6s}  "
                  f"{c.mean_hue*2:5.1f} {c.mean_sat:5.1f} {c.mean_val:5.1f} |")
        print("+-------------------------------------------------------------+\n")

        color_counts = Counter(c.color_label for c in coin_classifications)
        size_counts  = Counter(c.size_label  for c in coin_classifications)
        print("Color breakdown:", dict(color_counts))
        print("Size breakdown: ", dict(size_counts))

        if labels_dir is not None:
            base_name  = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            if os.path.isfile(label_path):
                h_img, w_img = image_bgr.shape[:2]
                ev = evaluate_detection(coin_classifications,
                                        label_path, w_img, h_img)
                print("\n  +---------------------------------------+")
                print("  |     GROUND-TRUTH EVALUATION           |")
                print("  +---------------------------------------+")
                print(f"  |  GT coins  : {ev['gt_count']:>3}                        |")
                print(f"  |  Detected  : {ev['det_count']:>3}                        |")
                print(f"  |  TP: {ev['tp']:>3}   FP: {ev['fp']:>3}   FN: {ev['fn']:>3}          |")
                print(f"  |  Precision : {ev['precision']:>6.1%}                   |")
                print(f"  |  Recall    : {ev['recall']:>6.1%}                   |")
                print("  +---------------------------------------+")
            else:
                print(f"[i] No label file found: {label_path}")

    annotated = draw_results(image_bgr, coin_classifications)

    base     = os.path.splitext(os.path.basename(image_path))[0]
    ann_path = os.path.join(output_dir, f"{base}_annotated.jpg")
    cv2.imwrite(ann_path, annotated)
    print(f"\n[OK] Annotated image -> {ann_path}")

    report_path = os.path.join(output_dir, f"{base}_report.png")
    build_report_figure(image_bgr, blurred, circles_filtered, annotated,
                        coin_classifications, report_path)

    print(f"\n-- SUMMARY {'-'*35}")
    print(f"  Total coins detected : {len(coin_classifications)}")
    print(f"{'-'*46}\n")

    return coin_classifications

def run_batch(images_dir, labels_dir=None, output_dir="batch_output",
             **pipeline_kwargs):

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not image_paths:
        print(f"[!] No images found in: {images_dir}")
        return []

    total = len(image_paths)
    print(f"\n[BATCH] {total} images  |  output -> {output_dir}\n")
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for idx, img_path in enumerate(image_paths, 1):
        fname = os.path.basename(img_path)
        print(f"[{idx:3d}/{total}] {fname}")
        try:
            coins = run_pipeline(
                img_path, output_dir=output_dir,
                labels_dir=labels_dir, **pipeline_kwargs
            )
            det = len(coins)
            gt = prec = rec = None
            if labels_dir:
                base = os.path.splitext(fname)[0]
                lp   = os.path.join(labels_dir, f"{base}.txt")
                if os.path.isfile(lp):
                    img_h, img_w = cv2.imread(img_path).shape[:2]
                    ev   = evaluate_detection(coins, lp, img_w, img_h)
                    gt, prec, rec = ev["gt_count"], ev["precision"], ev["recall"]
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            det = 0; gt = prec = rec = None

        records.append(dict(image=fname, det_count=det,
                            gt_count=gt, precision=prec, recall=rec))

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  BATCH SUMMARY  ({total} images)")
    print(f"{sep}")
    avg_det = sum(r["det_count"] for r in records) / total
    print(f"  Avg coins detected per image : {avg_det:.2f}")

    ev_recs = [r for r in records if r["precision"] is not None]
    if ev_recs:
        mean_prec  = sum(r["precision"] for r in ev_recs) / len(ev_recs)
        mean_rec   = sum(r["recall"]    for r in ev_recs) / len(ev_recs)
        exact_hits = sum(1 for r in ev_recs if r["det_count"] == r["gt_count"])
        print(f"  Mean Precision               : {mean_prec:.1%}")
        print(f"  Mean Recall                  : {mean_rec:.1%}")
        print(f"  Exact count accuracy         : "
              f"{exact_hits}/{len(ev_recs)} "
              f"({exact_hits/len(ev_recs):.1%})")
    print(f"{sep}\n")

    csv_path = os.path.join(output_dir, "batch_summary.csv")
    with open(csv_path, "w") as fh:
        fh.write("image,det_count,gt_count,precision,recall\n")
        for r in records:
            prec_s = f"{r['precision']:.4f}" if r["precision"] is not None else ""
            rec_s  = f"{r['recall']:.4f}"    if r["recall"]    is not None else ""
            gt_s   = str(r["gt_count"]) if r["gt_count"] is not None else ""
            fh.write(f"{r['image']},{r['det_count']},{gt_s},{prec_s},{rec_s}\n")
    print(f"[OK] Batch summary -> {csv_path}")
    return records

def main():
    parser = argparse.ArgumentParser(
        description="Coin Counting + Classification System (size & color features)")
    parser.add_argument("image",
                        help="Path to a single input image, "
                             "or to a folder of images when --batch is set.")
    parser.add_argument("--batch",             action="store_true",
                        help="Process every image in the folder given by 'image'.")
    parser.add_argument("--output-dir",        default="output")
    parser.add_argument("--labels-dir",        default=None,
                        help="Folder with YOLO .txt ground-truth files for evaluation")
    parser.add_argument("--dp",                type=float, default=1.2)
    parser.add_argument("--param1",            type=int,   default=80)
    parser.add_argument("--param2",            type=int,   default=35)
    parser.add_argument("--min-dist",          type=float, default=0.15)
    parser.add_argument("--min-r",             type=float, default=0.05)
    parser.add_argument("--max-r",             type=float, default=0.25)
    parser.add_argument("--edge-thresh",       type=float, default=30.0)
    parser.add_argument("--contrast-thresh",   type=float, default=2.0)
    parser.add_argument("--coherence-thresh",  type=float, default=0.65)
    args = parser.parse_args()

    pipeline_kw = dict(
        dp                  = args.dp,
        param1              = args.param1,
        param2              = args.param2,
        min_dist_factor     = args.min_dist,
        min_r_factor        = args.min_r,
        max_r_factor        = args.max_r,
        edge_threshold      = args.edge_thresh,
        contrast_threshold  = args.contrast_thresh,
        coherence_threshold = args.coherence_thresh,
    )

    if args.batch:
        run_batch(
            images_dir = args.image,
            labels_dir = args.labels_dir,
            output_dir = args.output_dir,
            **pipeline_kw,
        )
    else:
        run_pipeline(
            image_path = args.image,
            output_dir = args.output_dir,
            labels_dir = args.labels_dir,
            **pipeline_kw,
        )

if __name__ == "__main__":
    main()