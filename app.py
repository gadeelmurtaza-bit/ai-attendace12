import streamlit as st
import pandas as pd
import os
import tempfile
from PIL import Image
from face_utils import register_student, load_students, verify_face

st.set_page_config(page_title="AI Attendance System")
st.title("AI Attendance System – Cloud Compatible Version")

menu = ["Single Register", "Bulk Register", "Take Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

os.makedirs("data/images", exist_ok=True)

if not os.path.exists("data/students.csv"):
    pd.DataFrame(columns=["name", "roll", "image_path"]).to_csv("data/students.csv", index=False)


# ============ SINGLE REGISTER ============
if choice == "Single Register":
    name = st.text_input("Name")
    roll = st.text_input("Roll Number")
    img = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])

    if st.button("Register"):
        if name and roll and img:
            img_path = f"data/images/{roll}.jpg"
            image = Image.open(img)
            image.save(img_path)

            register_student(name, roll, img_path)
            st.success("Student Registered!")
        else:
            st.error("All fields required!")


# ============ BULK REGISTER ============
elif choice == "Bulk Register":
    file = st.file_uploader("Upload CSV (name,roll)", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.to_csv("data/students.csv", index=False)
        st.success("Bulk Registration Completed!")


# ============ TAKE ATTENDANCE ============
elif choice == "Take Attendance":
    st.header("Upload Live Photo for Attendance")

    students = load_students()

    if students.empty:
        st.warning("No students registered!")
    else:
        live = st.file_uploader("Upload Live Capture Photo", type=["jpg", "jpeg", "png"])

        if live:
            live_img = Image.open(live)
            st.image(live_img, caption="Uploaded Image")

            result = verify_face(live_img, students)
            st.success(result)
