import cv2
import numpy as np

img = cv2.imread("sample_image.png")

if img is None:
    print("Error: Image not found.")
    exit()



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (9, 9), 2)

edges = cv2.Canny(blur, 100, 200)

cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Edges", edges)

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