import cv2
import numpy as np
import sqlite3
import pickle
from datetime import datetime
import csv
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

embedder = FaceNet()
detector = MTCNN()



def load_students():
    connect = sqlite3.connect("students.db")

    cursor = connect.cursor()

    cursor.execute("SELECT class, section, roll, name, embedding FROM students")
    rows = cursor.fetchall()

    students = []
    for row in rows:
        class_name, section, roll, name, emb_blob = row
        embedding = pickle.loads(emb_blob)

        students.append({
            "class": class_name,
            "section": section,
            "roll": roll,
            "name": name,
            "embedding": embedding
        })

    connect.close()
    return students


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


def get_embedding(face_pixels):
    face_pixels = cv2.resize(face_pixels, (160, 160))
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = embedder.embeddings(face_pixels)[0]
    return embedding


def mark_attendance(student):

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{date_str}.csv"

    try:
        with open(filename, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Section", "Roll", "Name", "Time"])
    except:
        pass

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if (row[0] == student["class"] and
                row[1] == student["section"] and
                str(row[2]) == str(student["roll"])):
                return  

    now = datetime.now().strftime("%H:%M:%S")

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            student["class"],
            student["section"],
            student["roll"],
            student["name"],
            now
        ])

    print(f"Marked attendance for {student['name']} ({student['class']}-{student['section']}, Roll {student['roll']})")


def recognize_students():

    students = load_students()
    print(f"Loaded {len(students)} students from database.\n")

    cap = cv2.VideoCapture(0)
    print("Live Attendance System Started...")
    print("Press 'n' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detections = detector.detect_faces(rgb)

        for det in detections:
            x, y, w, h = det['box']
            x, y = abs(x), abs(y)
            face = rgb[y:y+h, x:x+w]

            embedding = get_embedding(face)


            best_match = None
            best_distance = 1

            for student in students:
                dist = cosine_distance(embedding, student["embedding"])
                if dist < best_distance:
                    best_distance = dist
                    best_match = student

            THRESHOLD = 0.35


            if best_distance < THRESHOLD:
                
                line1 = f"name: {best_match['name']}"
                line2 = f"class: {best_match['class']}({best_match['section']})"
                line3 = f"Roll: {best_match['roll']}"

                mark_attendance(best_match)
            else:
                line1 = "Unknown"
                line2 = ""
                line3 = ""

            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            cv2.putText(frame, line1, (x, y-45), font, font_scale, (0,255,0), thickness)
            cv2.putText(frame, line2, (x, y-25), font, font_scale, (0,255,0), thickness)
            cv2.putText(frame, line3, (x, y-5), font, font_scale, (0,255,0), thickness)

        cv2.imshow("Live Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

    cap.release()
    cv2.destroyAllWindows()

recognize_students()