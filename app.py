import os
# Suppress TensorFlow info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import sqlite3
import pickle
from datetime import datetime
import csv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
import base64

app = Flask(__name__)
CORS(app)

# Initialize models
embedder = FaceNet()
detector = MTCNN()

DB_NAME = "students.db"

def init_db():
    connect = sqlite3.connect(DB_NAME)
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
    connect.close()

init_db()

def get_embedding(face_pixels):
    face_pixels = cv2.resize(face_pixels, (160, 160))
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = embedder.embeddings(face_pixels)[0]
    return embedding

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def load_students():
    connect = sqlite3.connect(DB_NAME)
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

def decode_image(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    try:
        class_name = data.get('class')
        section = data.get('section')
        roll = data.get('roll')
        name = data.get('name')
        image_data = data.get('image')

        if not all([class_name, section, roll, name, image_data]):
            return jsonify({"error": "Missing fields"}), 400

        img = decode_image(image_data)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        if not detections:
            return jsonify({"error": "No face detected"}), 400

        # Get the largest face
        det = max(detections, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = det['box']
        x, y = abs(x), abs(y)
        face = rgb[y:y+h, x:x+w]
        
        embedding = get_embedding(face)
        emb_blob = pickle.dumps(embedding)

        connect = sqlite3.connect(DB_NAME)
        cursor = connect.cursor()
        try:
            cursor.execute(
                "INSERT INTO students (class, section, roll, name, embedding) VALUES (?, ?, ?, ?, ?)",
                (class_name, section, roll, name, emb_blob)
            )
            connect.commit()
        except sqlite3.IntegrityError:
            connect.close()
            return jsonify({"error": "Student already exists"}), 409
        
        connect.close()
        return jsonify({"message": "Student registered successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        img = decode_image(image_data)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        if not detections:
            return jsonify({"message": "No face detected"}), 200

        students = load_students()
        recognized_students = []

        for det in detections:
            x, y, w, h = det['box']
            x, y = abs(x), abs(y)
            face = rgb[y:y+h, x:x+w]
            embedding = get_embedding(face)

            best_match = None
            min_dist = 1.0 # Threshold

            for student in students:
                dist = cosine_distance(embedding, student["embedding"])
                if dist < 0.5: # Threshold for recognition
                    if dist < min_dist:
                        min_dist = dist
                        best_match = student

            if best_match:
                mark_attendance(best_match)
                recognized_students.append({
                    "name": best_match["name"],
                    "class": best_match["class"],
                    "section": best_match["section"],
                    "roll": best_match["roll"],
                    "confidence": float(1 - min_dist)
                })

        return jsonify({"recognized": recognized_students}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def mark_attendance(student):
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{date_str}.csv"
    
    file_exists = os.path.isfile(filename)

    try:
        if not file_exists:
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Class", "Section", "Roll", "Name", "Time"])

        # Check if already marked
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if (row[0] == student["class"] and
                    row[1] == student["section"] and
                    str(row[2]) == str(student["roll"])):
                    return # Already marked

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
    except Exception as e:
        print(f"Error marking attendance: {e}")

@app.route('/attendance', methods=['GET'])
def get_attendance():
    date_str = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    filename = f"attendance_{date_str}.csv"
    
    if not os.path.exists(filename):
        return jsonify({"date": date_str, "records": []}), 200

    records = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    return jsonify({"date": date_str, "records": records}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
