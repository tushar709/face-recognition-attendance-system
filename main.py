import cv2
import pandas as pd
from datetime import datetime

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("trainer/classifier.xml")

names = {
    1: "Tushar",
    2: "vivek",
    # 3: "Priya"
}

def mark_attendance(name):

    now = datetime.now()

    time = now.strftime("%H:%M:%S")
    date = now.strftime("%d-%m-%Y")

    data = {
        "Name": [name],
        "Time": [time],
        "Date": [date]
    }

    df = pd.DataFrame(data)

    try:
        old = pd.read_csv("Attendance.csv")
        df = pd.concat([old, df], ignore_index=True)
    except:
        pass

    df.to_csv("Attendance.csv", index=False)

cap = cv2.VideoCapture(0)

marked = []

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x,y,w,h) in faces:

        id, pred = clf.predict(gray[y:y+h, x:x+w])

        confidence = int(100 * (1 - pred / 300))

        if confidence > 75:

            name = names[id]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.putText(
                frame,
                name,
                (x,y-10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255,255,255),
                2
            )

            if name not in marked:
                mark_attendance(name)
                marked.append(name)

        else:

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

            cv2.putText(
                frame,
                "Unknown",
                (x,y-10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255,255,255),
                2
            )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()