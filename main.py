import cv2

detect_faces = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
cam = cv2.VideoCapture(0)
count = 0
while True:
    OK, frame = cam.read()
    faces = detect_faces.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y: y + h, x: x + w], (100, 100))
        cv2.imwrite('img/anh_{}.jpg'.format(count), roi)
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 1)

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()