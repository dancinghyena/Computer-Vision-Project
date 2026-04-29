
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Step 1 – Pre-processing

def preprocess(image_bgr):
    """Convert to greyscale, apply CLAHE for contrast, and blur heavily
    to suppress background texture while preserving strong coin edges."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE normalises local contrast so coins stand out from any background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Heavy Gaussian blur kills texture noise (carpet fibres, gravel, etc.)
    blurred = cv2.GaussianBlur(enhanced, (15, 15), 3)
    return gray, blurred


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Circular Hough Transform
# ──────────────────────────────────────────────────────────────────────────────
def detect_circles(blurred, dp=1.2, min_dist_factor=0.15,
                   param1=80, param2=35, min_r_factor=0.05, max_r_factor=0.35):
    """Run HoughCircles on the blurred greyscale image."""
    h, w = blurred.shape
    min_dist  = int(w * min_dist_factor)
    min_r     = int(w * min_r_factor)
    max_r     = int(w * max_r_factor)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    return circles  # shape (1, N, 3) or None


# ──────────────────────────────────────────────────────────────────────────────
# Step 2b – Post-detection filtering  (reject false positives)
# ──────────────────────────────────────────────────────────────────────────────
def filter_circles(circles_raw, gray_image, edge_threshold=30.0,
                    contrast_threshold=2.0, coherence_threshold=0.65):

    if circles_raw is None:
        return None

    # Pre-compute gradient components
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    h, w = gray_image.shape
    kept = []

    # Create coordinate grids for mask operations
    yy, xx = np.ogrid[:h, :w]

    for cx, cy, r in circles_raw[0]:
        cx, cy, r = float(cx), float(cy), float(r)
        ri = int(round(r))
        if ri < 5:
            continue

        # ── Test 1: Edge strength along perimeter ────────────────────────
        n_samples = max(36, int(2 * np.pi * ri * 0.3))
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[valid]
        ys = ys[valid]
        angles_v = angles[valid]

        if len(xs) < n_samples * 0.5:
            continue  # too much of the perimeter is outside the image

        mean_edge = grad_mag[ys, xs].mean()
        if mean_edge < edge_threshold:
            continue

        # ── Test 2: Radial gradient coherence ────────────────────────────
        # For each perimeter point, check if the gradient direction aligns
        # with the radial direction (centre → point).
        gx_vals = grad_x[ys, xs]
        gy_vals = grad_y[ys, xs]
        # Radial unit vectors (centre → perimeter point)
        rx = np.cos(angles_v)
        ry = np.sin(angles_v)
        # Dot product of gradient with radial direction, normalised
        dot = gx_vals * rx + gy_vals * ry
        g_mags = grad_mag[ys, xs]
        # Avoid division by zero for very small gradients
        nonzero = g_mags > 1.0
        if nonzero.sum() < n_samples * 0.3:
            continue
        cos_angle = np.abs(dot[nonzero]) / g_mags[nonzero]
        # cos(45°) ≈ 0.707; count how many are within 45° of radial
        coherence = (cos_angle > 0.707).sum() / float(nonzero.sum())
        if coherence < coherence_threshold:
            continue

        # ── Test 3: Inside vs outside contrast ───────────────────────────
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        inner_mask = dist_sq <= (0.85 * r) ** 2          # inner 85 % of radius
        outer_mask = (dist_sq > (1.1 * r) ** 2) & \
                     (dist_sq <= (1.4 * r) ** 2)          # annulus just outside

        inner_pixels = gray_image[inner_mask]
        outer_pixels = gray_image[outer_mask]

        if inner_pixels.size < 10 or outer_pixels.size < 10:
            continue

        contrast = abs(float(inner_pixels.mean()) - float(outer_pixels.mean()))
        if contrast < contrast_threshold:
            continue

        kept.append([cx, cy, r, mean_edge])  # keep edge score for tie-breaking

    if not kept:
        return None

    # ── Test 3: Non-maximum suppression for overlapping circles ──────────
    # When two circles overlap substantially, keep only the one with
    # stronger edge evidence.  This removes concentric false detections.
    kept.sort(key=lambda c: c[3], reverse=True)  # strongest edge first
    final = []
    for cand in kept:
        cx1, cy1, r1, _ = cand
        suppress = False
        for accepted in final:
            cx2, cy2, r2, _ = accepted
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            # If centres are within the larger radius, they overlap a lot
            if dist < max(r1, r2) * 0.7:
                suppress = True
                break
        if not suppress:
            final.append(cand)

    if not final:
        return None

    # Strip the edge score, keep only (cx, cy, r)
    final3 = [[c[0], c[1], c[2]] for c in final]
    return np.array([[final3]], dtype=np.float32).reshape(1, -1, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def draw_results(image_bgr, circles):
    """Annotate the image with simple circles and numbers."""
    output = image_bgr.copy()
    bgr_col = (0, 255, 0) # Green

    for i, (cx, cy, r) in enumerate(circles, 1):
        cx, cy, r = int(cx), int(cy), int(r)

        # Outer circle
        cv2.circle(output, (cx, cy), r, bgr_col, 3)
        # Centre dot
        cv2.circle(output, (cx, cy), 4, bgr_col, -1)

        # Label background + text
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, r / 60)
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = cx - tw // 2
        ty = cy + th // 2

        cv2.rectangle(output, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(output, text, (tx, ty), font, font_scale, bgr_col, thickness, cv2.LINE_AA)

    return output


def build_report_figure(image_bgr, blurred, circles_raw, annotated, num_detected, output_path):
    """Produce a simplified 2x2 matplotlib figure with diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    title_kw  = dict(color="white", fontsize=12, fontweight="bold", pad=8)

    # ── Panel 1: Original image
    axes[0, 0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1 · Original Image", **title_kw)
    axes[0, 0].axis("off")

    # ── Panel 2: Greyscale + blur
    axes[0, 1].imshow(blurred, cmap="gray")
    axes[0, 1].set_title("2 · Blurred Image", **title_kw)
    axes[0, 1].axis("off")

    # ── Panel 3: Hough circles overlay
    hough_vis = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    if circles_raw is not None:
        for x, y, r in np.uint16(np.around(circles_raw[0])):
            cv2.circle(hough_vis, (x, y), r, (0, 220, 120), 2)
            cv2.circle(hough_vis, (x, y), 3, (255, 80, 80), -1)
    axes[1, 0].imshow(hough_vis)
    axes[1, 0].set_title(f"3 · Hough Circles (n={num_detected})", **title_kw)
    axes[1, 0].axis("off")

    # ── Panel 4: Annotated result
    axes[1, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"4 · Counting Result: {num_detected} Coins", **title_kw)
    axes[1, 1].axis("off")

    fig.suptitle(f"Coin Counting System — Total: {num_detected}", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Report saved -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(image_path, output_dir="output",
                 dp=1.2, param1=80, param2=35,
                 min_dist_factor=0.15, min_r_factor=0.05, max_r_factor=0.25,
                 edge_threshold=30.0, contrast_threshold=2.0, coherence_threshold=0.65):

    os.makedirs(output_dir, exist_ok=True)

    # --- Load ---
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    print(f"[OK] Loaded {image_path}  ({image_bgr.shape[1]}x{image_bgr.shape[0]})")

    # --- Pre-process ---
    gray, blurred = preprocess(image_bgr)

    # --- Hough ---
    circles_raw = detect_circles(blurred, dp=dp,
                                  min_dist_factor=min_dist_factor,
                                  param1=param1, param2=param2,
                                  min_r_factor=min_r_factor,
                                  max_r_factor=max_r_factor)

    # --- Edge-strength + contrast filter to reject false positives ---
    circles_filtered = filter_circles(circles_raw, gray,
                                       edge_threshold=edge_threshold,
                                       contrast_threshold=contrast_threshold,
                                       coherence_threshold=coherence_threshold)

    n_before = 0 if circles_raw is None else len(circles_raw[0])
    detected_list = []
    if circles_filtered is None:
        if n_before > 0:
            print(f"[!] Hough found {n_before} candidate(s) but all were rejected by edge filter.")
        else:
            print("[!] No circles detected. Try adjusting Hough parameters.")
    else:
        detected_list = circles_filtered[0]
        print(f"[OK] Hough candidates: {n_before} -> after edge filter: {len(detected_list)} circle(s)")
        for cx, cy, r in detected_list:
            print(f"   Circle at ({int(cx):4d},{int(cy):4d})  r={int(r)}px")

    # --- Draw ---
    annotated = draw_results(image_bgr, detected_list)

    # --- Save annotated ---
    base      = os.path.splitext(os.path.basename(image_path))[0]
    ann_path  = os.path.join(output_dir, f"{base}_annotated.jpg")
    cv2.imwrite(ann_path, annotated)
    print(f"[OK] Annotated image -> {ann_path}")

    # --- Report figure ---
    report_path = os.path.join(output_dir, f"{base}_report.png")
    build_report_figure(image_bgr, blurred, circles_filtered, annotated, len(detected_list), report_path)

    # --- Summary ---
    print("\n-- SUMMARY ----------------------------------")
    print(f"  Total coins : {len(detected_list)}")
    print("---------------------------------------------\n")

    return detected_list


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Coin Counting System (Simplified)")
    parser.add_argument("image",          help="Path to input image")
    parser.add_argument("--output-dir",   default="output")
    parser.add_argument("--dp",           type=float, default=1.2)
    parser.add_argument("--param1",       type=int,   default=50)
    parser.add_argument("--param2",       type=int,   default=25)
    parser.add_argument("--min-dist",     type=float, default=0.15,
                        help="Min inter-centre dist as fraction of image width")
    parser.add_argument("--min-r",        type=float, default=0.05,
                        help="Min radius as fraction of image width")
    parser.add_argument("--max-r",        type=float, default=0.25,
                        help="Max radius as fraction of image width")
    parser.add_argument("--edge-thresh",  type=float, default=30.0,
                        help="Min mean edge strength along circle perimeter")
    parser.add_argument("--contrast-thresh", type=float, default=0.5,
                        help="Min intensity contrast between circle inside and outside")
    args = parser.parse_args()

    run_pipeline(
        image_path         = args.image,
        output_dir         = args.output_dir,
        dp                 = args.dp,
        param1             = args.param1,
        param2             = args.param2,
        min_dist_factor    = args.min_dist,
        min_r_factor       = args.min_r,
        max_r_factor       = args.max_r,
        edge_threshold     = args.edge_thresh,
        contrast_threshold = args.contrast_thresh,
    )

if __name__ == "__main__":
    main()
