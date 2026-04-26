import cv2
import numpy as np
import kagglehub
import os

path = kagglehub.dataset_download("balabaskar/count-coins-image-dataset")
print("Dataset location:", path)

# 2. Point to a specific image in that dataset
# The dataset path usually contains subfolders like 'train' or 'test'
# Search recursively for images
image_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

if not image_files:
    print("Error: No images found in dataset")
    exit()

# Display available images
print(f"\nFound {len(image_files)} images:")
for idx, img_file in enumerate(image_files[:10]):  # Show first 10
    print(f"{idx}: {os.path.basename(img_file)}")
if len(image_files) > 10:
    print(f"... and {len(image_files) - 10} more")

# Get user input
while True:
    try:
        image_order = int(input(f"\nEnter image number (0-{len(image_files)-1}): "))
        if 0 <= image_order < len(image_files):
            break
        else:
            print(f"Invalid number. Please enter a number between 0 and {len(image_files)-1}")
    except ValueError:
        print("Please enter a valid integer")

img_path = image_files[image_order] 
print(f"Using image: {img_path}")

img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image at {img_path}")
    exit()



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (9, 9), 2)

edges = cv2.Canny(blur, 100, 200)

#cv2.imshow("Original", img)
#cv2.imshow("Gray", gray)
#cv2.imshow("Blur", blur)
#cv2.imshow("Edges", edges)




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

output = img.copy()
count = 0

if circles is not None:
    circles = np.uint16(np.around(circles))

    for (x, y, r) in circles[0]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # center point
        count += 1

cv2.putText(output, f"Coins: {count}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Detected Coins", output)

cv2.waitKey(0)
cv2.destroyAllWindows()