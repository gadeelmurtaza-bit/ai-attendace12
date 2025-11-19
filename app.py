import streamlit as st
import os
import cv2
import pandas as pd
from face_utils import register_student, load_known_faces, mark_attendance
import tempfile

st.set_page_config(page_title="AI Attendance System")
st.title("AI Attendance System Using Face Recognition")

menu = ["Bulk Register", "Single Register", "Take Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

# Ensure data folders exist
os.makedirs("data/images", exist_ok=True)
if not os.path.exists("data/students.csv"):
    pd.DataFrame(columns=["name", "roll", "image_path"]).to_csv("data/students.csv", index=False)


# BULK REGISTER
if choice == "Bulk Register":
    st.header("Bulk Student Registration")
    file = st.file_uploader("Upload CSV with columns: name, roll", type=["csv"])
    
    if file:
        df = pd.read_csv(file)
        df.to_csv("data/students.csv", index=False)
        st.success("Bulk registration completed!")


# SINGLE REGISTER
elif choice == "Single Register":
    st.header("Single Student Registration")

    name = st.text_input("Student Name")
    roll = st.text_input("Roll Number")
    img = st.file_uploader("Upload Student Photo (JPG)", type=["jpg", "jpeg", "png"])

    if st.button("Register Student"):
        if name and roll and img:
            # save uploaded image
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(img.read())
            save_path = f"data/images/{roll}.jpg"
            os.rename(temp.name, save_path)

            register_student(name, roll, save_path)
            st.success("Student Registered Successfully!")
        else:
            st.error("Please fill all fields!")


# LIVE ATTENDANCE
elif choice == "Take Attendance":
    st.header("Live Attendance System")

    known_encodings, known_data = load_known_faces()

    if len(known_encodings) == 0:
        st.warning("No registered student found!")
    else:
        st.info("Click Start to open camera")

        if st.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                status, frame = mark_attendance(frame, known_encodings, known_data)

                stframe.image(frame, channels="BGR")
                st.write(status)

                if st.button("Stop Camera"):
                    break

            cap.release()
