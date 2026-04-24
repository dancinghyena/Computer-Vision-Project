import cv2

img = cv2.imread("sample_image.png")

if img is None:
    print("Error: Image not found.")
    exit()


cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()    
