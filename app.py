import streamlit as st
import pandas as pd
import cv2
import os
import tempfile
from deepface import DeepFace
from face_utils import register_student, load_registered_students, match_face

st.set_page_config(page_title="AI Attendance System")
st.title("AI Attendance System – DeepFace Version")

menu = ["Bulk Register", "Single Register", "Take Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

os.makedirs("data/images", exist_ok=True)

if not os.path.exists("data/students.csv"):
    pd.DataFrame(columns=["name", "roll", "image_path"]).to_csv("data/students.csv", index=False)


# Bulk register
if choice == "Bulk Register":
    file = st.file_uploader("Upload CSV with name, roll", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.to_csv("data/students.csv", index=False)
        st.success("Bulk registration completed!")


# Single register
elif choice == "Single Register":
    name = st.text_input("Student Name")
    roll = st.text_input("Roll Number")
    img = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])

    if st.button("Register"):
        if name and roll and img:
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(img.read())
            save_path = f"data/images/{roll}.jpg"
            os.rename(temp.name, save_path)
            register_student(name, roll, save_path)
            st.success("Student registered successfully!")
        else:
            st.error("All fields required!")


# Take attendance
elif choice == "Take Attendance":
    st.subheader("Live Attendance System")

    students = load_registered_students()
    if students.empty:
        st.warning("No students registered!")
    else:
        if st.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            frame_place = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera not found!")
                    break

                result, frame = match_face(frame, students)

                frame_place.image(frame, channels="BGR")
                st.write(result)

                stop = st.button("Stop Camera")
                if stop:
                    break

            cap.release()
