import cv2
import sqlite3
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle


embedder = FaceNet()
detector = MTCNN()

connect = sqlite3.connect("students.db")
cursor = connect.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class TEXT,
    section TEXT,
    roll INTEGER,
    name TEXT,
    embedding BLOB,
    UNIQUE(class, section, roll)
)
""")

connect.commit()


def get_embedding(face_pixels):
    face_pixels = cv2.resize(face_pixels, (160, 160))
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = embedder.embeddings(face_pixels)[0]
    return embedding


def register_student():
    print("\n..... Student Registration ......")

    class_name = input("Enter Class (e.g., 1, 12, LKG, Nursery, A1): ").strip()
    section = input("Enter Section (A-Z): ").strip().upper()
    roll = input("Enter Roll Number (Number): ").strip()
    name = input("Enter Student Name: ")

    
    if not (len(section) == 1 and section.isalpha()):
        print("\n Section must be a single letter A–Z")
        return

    
    if not roll.isdigit():
        print("\n Roll number must be numeric")
        return

    roll = int(roll)

    
    cursor.execute(
        "SELECT * FROM students WHERE class=? AND section=? AND roll=?",
        (class_name, section, roll)
    )
    if cursor.fetchone():
        print("\n A student with this Class + Section + Roll already exists!")
        return


    cap = cv2.VideoCapture(0)
    print("\n Align face → Press 'm' to capture → Press 'n' to quit\n")

    captured_face = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

    
        for det in detections:
            x, y, w, h = det['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Register Student - Webcam", frame)

        key = cv2.waitKey(1)

        if key == ord('m'):
            if len(detections) == 0:
                print("❌ No face detected! Try again.")
                continue

            x, y, w, h = detections[0]['box']
            x, y = abs(x), abs(y)
            captured_face = rgb[y:y+h, x:x+w]
            print("✅ Face captured!")
            break

        elif key == ord('n'):
            print(" Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    embedding = get_embedding(captured_face)


    cursor.execute("""
        INSERT INTO students (class, section, roll, name, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (
        class_name,
        section,
        roll,
        name,
        pickle.dumps(embedding)
    ))

    connect.commit()

    print("\nStudent Registered Successfully!")
    print(f"Class: {class_name}")
    print(f"Section: {section}")
    print(f"Roll: {roll}")
    print(f"Name: {name}\n")

register_student()