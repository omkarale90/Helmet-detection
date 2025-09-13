import cv2
img=cv2.imread('eagle.jpg')
#resizing image
resized=cv2.resize(img,(300,300))
# cv2.imshow("eagle",img)

gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(resized,cv2.COLOR_BGR2HSV)

#threshhold image
gray_for_bw = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
__,bw=cv2.threshold(gray_for_bw,127,255,cv2.THRESH_BINARY)

cv2.imshow("original image",resized)
cv2.imshow("gray image",gray)
cv2.imshow('hsv image',hsv)
cv2.imshow('gray_for_bw',bw)
cv2.waitKey(0)
cv2.destroyAllWindows()