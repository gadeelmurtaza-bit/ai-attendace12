import pandas as pd
from deepface import DeepFace
import cv2

STUDENT_CSV = "data/students.csv"


def register_student(name, roll, image_path):
    df = pd.read_csv(STUDENT_CSV)
    df.loc[len(df)] = [name, roll, image_path]
    df.to_csv(STUDENT_CSV, index=False)


def load_registered_students():
    return pd.read_csv(STUDENT_CSV)


def match_face(frame, students):
    result = "No Match Found"

    # Save frame temporarily
    cv2.imwrite("temp.jpg", frame)

    for _, row in students.iterrows():
        try:
            verified = DeepFace.verify("temp.jpg", row["image_path"], enforce_detection=False)

            if verified["verified"]:
                name = row["name"]
                roll = row["roll"]
                result = f"Present: {name} ({roll})"

                # Draw green box
                cv2.putText(frame, name, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                break

        except:
            continue

    return result, frame
