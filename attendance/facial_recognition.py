import face_recognition
from PIL import Image
import numpy as np
from .models import Student,AttendanceRecord
import datetime

def recognize_faces(image):
    # Convert uploaded image to PIL format
    uploaded_image = Image.open(image)
    
    # Convert image to RGB mode (if it's not already)
    if uploaded_image.mode != 'RGB':
        uploaded_image = uploaded_image.convert('RGB')
    
    # Convert PIL image to numpy array
    uploaded_image_np = np.array(uploaded_image)
    
    known_face_encodings = []
    known_face_names = []

    students = Student.objects.all()
    for student in students:
        student_image = face_recognition.load_image_file(student.image.path)
        student_face_encoding = face_recognition.face_encodings(student_image)[0]
        known_face_encodings.append(student_face_encoding)
        known_face_names.append(student.name)

    unknown_face_encodings = face_recognition.face_encodings(uploaded_image_np)

    results = []
    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        results.append(name)

    return results
