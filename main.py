import cv2
import numpy as np
import kagglehub
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time

# Download dataset from Kaggle (will be cached after first download)
print("Downloading/accessing dataset from Kaggle...")
path = kagglehub.dataset_download("balabaskar/count-coins-image-dataset")
print("Dataset location:", path)

# Load ground truth labels
csv_path = os.path.join(path, "coins_count_values.csv")
df_labels = pd.read_csv(csv_path)
print(f"Loaded {len(df_labels)} labeled images\n")

# Create a mapping from image filename to true coin count
label_dict = {}
for _, row in df_labels.iterrows():
    img_path = os.path.join(path, "coins_images", "coins_images", row['folder'], row['image_name'])
    label_dict[img_path] = row['coins_count']
    
# Get all images that have labels
image_files = [img_path for img_path in label_dict.keys() if os.path.exists(img_path)]
print(f"Found {len(image_files)} images with ground truth labels\n")

# Process all images
total_time = 0
total_coins = 0
predicted_counts = []
true_counts = []

for idx, img_path in enumerate(image_files, 1):
    true_count = label_dict[img_path]
    true_counts.append(true_count)
    
    if idx % 100 == 0 or idx == len(image_files):
        print(f"Processing: {idx}/{len(image_files)} images...")
    
    start_time = time.time()
    
    img = cv2.imread(img_path)

    if img is None:
        continue


    # Resize so your radius guesses actually mean something
    height, width = img.shape[:2]
    target_width = 800
    aspect_ratio = height / width
    img = cv2.resize(img, (target_width, int(target_width * aspect_ratio)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    edges = cv2.Canny(blur, 100, 200)

    circles = cv2.HoughCircles(
        blur,                    # use blurred grayscale image
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,              # distance between coin centers
        param1=100,              # Canny high threshold (internal)
        param2=30,               # detection sensitivity
        minRadius=20,            # adjust based on your coins
        maxRadius=100
    )

    count = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0])

    predicted_counts.append(count)
    elapsed_time = time.time() - start_time
    total_time += elapsed_time
    total_coins += count

print(f"\n{'='*60}")
print(f"ACCURACY METRICS ON ENTIRE DATASET")
print(f"{'='*60}")

# Calculate accuracy metrics
mae = mean_absolute_error(true_counts, predicted_counts)
rmse = np.sqrt(np.mean((np.array(predicted_counts) - np.array(true_counts))**2))

# Exact match accuracy
exact_match = np.sum(np.array(predicted_counts) == np.array(true_counts))
exact_match_pct = (exact_match / len(true_counts)) * 100

# Off-by-one accuracy
off_by_one = np.sum(np.abs(np.array(predicted_counts) - np.array(true_counts)) <= 1)
off_by_one_pct = (off_by_one / len(true_counts)) * 100

print(f"Total images processed: {len(image_files)}")
print(f"Total coins detected: {total_coins}")
print(f"Total true coins: {sum(true_counts)}")
print()
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Exact Match Accuracy: {exact_match}/{len(true_counts)} ({exact_match_pct:.1f}%)")
print(f"Off-by-One Accuracy: {off_by_one}/{len(true_counts)} ({off_by_one_pct:.1f}%)")
print()
print(f"Average coins per image (predicted): {total_coins/len(image_files):.1f}")
print(f"Average coins per image (true): {sum(true_counts)/len(true_counts):.1f}")
print(f"Total processing time: {total_time:.2f}s")
print(f"Average time per image: {total_time/len(image_files):.3f}s")
print(f"{'='*60}")