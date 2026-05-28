# Computer Vision Project — Coin Detection & Classification

A Python computer-vision pipeline that detects circular coins in images, classifies them by **size** and **color**, and optionally evaluates detections against YOLO-format ground-truth labels.

Built with OpenCV (Hough circle transform), NumPy, and Matplotlib.

## Features

- **Preprocessing** — CLAHE contrast enhancement, Gaussian blur, automatic white balance
- **Circle detection** — `cv2.HoughCircles` with tunable sensitivity parameters
- **False-positive filtering** — edge strength, gradient coherence, and inner/outer contrast checks
- **Classification** — per-coin size percentile labels and HSV/LAB color labels
- **Visualization** — annotated output images saved to disk
- **Batch mode** — process a folder of images and export `batch_summary.csv`
- **Evaluation** — optional precision/recall vs. YOLO `.txt` label files (circle IoU matching)

## Requirements

- Python 3.8+
- [OpenCV](https://opencv.org/) (`opencv-python`)
- NumPy
- Matplotlib

```bash
pip install opencv-python numpy matplotlib
```

## Usage

### Single image

```bash
python main.py path/to/image.jpg
```

Annotated results are written to the `output/` directory (default).

### Batch processing

```bash
python main.py path/to/images/ --batch --output-dir output --labels-dir path/to/labels/
```

When `--labels-dir` is provided, each image is evaluated against matching YOLO label files (`<image_stem>.txt`), and a summary CSV is generated.

### Useful CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `output` | Directory for annotated images and CSV |
| `--labels-dir` | — | Folder of YOLO ground-truth `.txt` files |
| `--dp` | `1.2` | Hough accumulator resolution |
| `--param1` / `--param2` | `80` / `35` | Canny / accumulator threshold |
| `--min-dist` | `0.15` | Min center distance (fraction of image width) |
| `--min-r` / `--max-r` | `0.05` / `0.25` | Radius bounds (fraction of width) |
| `--edge-thresh` | `30.0` | Minimum mean edge strength on circle boundary |
| `--contrast-thresh` | `2.0` | Min inner/outer luminance contrast |
| `--coherence-thresh` | `0.65` | Min gradient-direction coherence |

## Project structure

```
Computer-Vision-Project/
├── main.py                 # Full pipeline, CLI, batch runner
├── archive_2/              # Archived images / experiments
├── final_batch_results/    # Saved batch run outputs
└── README.md
```

## Pipeline overview

```
Input image
    → white balance & preprocess (CLAHE + blur)
    → Hough circle detection
    → geometric / photometric filtering
    → size & color feature extraction
    → annotated output + optional evaluation metrics
```

## License

Academic / coursework project. Use and modify as needed with attribution.
