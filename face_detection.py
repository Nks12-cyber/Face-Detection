import cv2

img = cv2.imread('1.jpg')
trained_faces = cv2.CascadeClassifier('trained.xml')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_codinate = trained_faces.detectMultiScale(gray_img)
for x, y, w, h in face_codinate:
    cv2.rectangle(img, (x, y), (x+w, y+h), (23, 255, 65), 2)
cv2.imshow('testing', img)
cv2.waitKey()


