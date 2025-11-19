import face_recognition
import pandas as pd
import cv2

STUDENT_CSV = "data/students.csv"

# Register student
def register_student(name, roll, image_path):
    df = pd.read_csv(STUDENT_CSV)
    df.loc[len(df)] = [name, roll, image_path]
    df.to_csv(STUDENT_CSV, index=False)


# Load all known faces
def load_known_faces():
    df = pd.read_csv(STUDENT_CSV)
    encodings = []
    metadata = []

    for _, row in df.iterrows():
        image = face_recognition.load_image_file(row["image_path"])
        face_enc = face_recognition.face_encodings(image)
        if face_enc:
            encodings.append(face_enc[0])
            metadata.append({"name": row["name"], "roll": row["roll"]})

    return encodings, metadata


# Attendance marking
attendance_done = set()

def mark_attendance(frame, known_encodings, metadata):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locations)

    result = "No face detected"

    for enc, loc in zip(encs, locations):
        matches = face_recognition.compare_faces(known_encodings, enc)

        if True in matches:
            idx = matches.index(True)
            name = metadata[idx]["name"]
            roll = metadata[idx]["roll"]

            result = f"Present: {name} ({roll})"
            attendance_done.add(roll)

            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        else:
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

    return result, frame
