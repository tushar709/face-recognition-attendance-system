import cv2
import os

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

def face_cropped(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        1.3,
        5
    )

    if len(faces) == 0:
        return None

    for (x,y,w,h) in faces:
        face_cropped = img[y:y+h, x:x+w]

    return face_cropped

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

id = input("Enter User ID: ")
name = input("Enter Name: ")

count = 0

if not os.path.exists("dataset"):
    os.makedirs("dataset")

while True:

    ret, frame = cap.read()

    if face_cropped(frame) is not None:

        count += 1

        face = cv2.resize(face_cropped(frame), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name = f"dataset/user.{id}.{count}.jpg"

        cv2.imwrite(file_name, face)

        cv2.putText(
            face,
            str(count),
            (50,50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Cropped Face", face)

    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()

print("Face samples collected successfully")