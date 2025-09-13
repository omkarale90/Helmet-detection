import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img=cv2.imread('omk.jpg')


resized_img=cv2.resize(img,(400,400))
gray=cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,1.1,5, minSize=(30,30))


for (x,y,w,h) in faces:
    cv2.rectangle(resized_img,(x,y),(x+w,y+h),(255,0,0),2)

    roi_gray=gray[y:y+h,x:x+w]
    roi_color=resized_img[y:y+h,x:x+w]
    eyes=eye_cascade.detectMultiScale(roi_gray,1.1,10, minSize=(15,15))
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv2.imshow('face & eye detection', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
