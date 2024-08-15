
#from collections import defaultdict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import cv2
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect, get_object_or_404
#import numpy as np
from .models import Student, AttendanceRecord
import datetime
from datetime import timedelta
from django.views import View
from pathlib import Path
from .forms import CustomLoginForm, StudentForm, AttendanceForm
from deepface import DeepFace
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth import authenticate, login,logout
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
#from django.contrib.auth.forms import AuthenticationForm

def send_attendance_alert(student_email, subject, message):
    send_mail(
        subject,
        message,
        settings.EMAIL_HOST_USER,
        [student_email],
        fail_silently=False,
    )

def index(request):
    return render(request, 'attendance/index.html')

def adminn(request):
    return render(request, 'attendance/adminn.html')
@csrf_exempt
def loginn(request):
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('adminn')  # Redirect to the admin index page
    else:
        form = CustomLoginForm()
    return render(request, 'attendance/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('index')

def student_list(request):
    students = Student.objects.all()
    return render(request, 'attendance/student_list.html', {'students': students})

def attendance_list(request):
    records = AttendanceRecord.objects.all()
    return render(request, 'attendance/attendance_list.html', {'records': records})

def upload_camera(request):
    return render(request, 'attendance/upload_image.html')

class StudentListView(View):
    def get(self, request):
        students = Student.objects.all()
        return render(request, 'attendance/student_list.html', {'students': students})

class StudentCreateView(View):
    def get(self, request):
        form = StudentForm()
        return render(request, 'students/student_form.html', {'form': form})

    def post(self, request):
        form = StudentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('student-list')
        else:
            print(form.errors)
        return render(request, 'students/student_form.html', {'form': form})

class StudentUpdateView(View):
    def get(self, request, pk):
        student = get_object_or_404(Student, pk=pk)
        form = StudentForm(instance=student)
        return render(request, 'students/student_form.html', {'form': form})

    def post(self, request, pk):
        student = get_object_or_404(Student, pk=pk)
        form = StudentForm(request.POST, request.FILES, instance=student)
        if form.is_valid():
            form.save()
            return redirect('student-list')
        else:
            print(form.errors)
        return render(request, 'students/student_form.html', {'form': form})

class StudentDeleteView(View):
    def post(self, request, pk):
        student = get_object_or_404(Student, pk=pk)
        print(f"Deleting student: {student.name}")
        student.delete()
        return redirect('student-list')

class AttendanceListView(View):
    def get(self, request):
        attendance_records = AttendanceRecord.objects.all()
        return render(request, 'attendance/attendance_list.html', {'attendance_records': attendance_records})

class AttendanceCreateView(View):
    def get(self, request):
        form = AttendanceForm()
        return render(request, 'attendance/attendance_form.html', {'form': form})

    def post(self, request):
        form = AttendanceForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('attendance_list')
        return render(request, 'attendance/attendance_form.html', {'form': form})

class AttendanceEditView(View):
    def get(self, request, pk):
        attendance = get_object_or_404(AttendanceRecord, pk=pk)
        form = AttendanceForm(instance=attendance)
        return render(request, 'attendance/attendance_form.html', {'form': form})

    def post(self, request, pk):
        attendance = get_object_or_404(AttendanceRecord, pk=pk)
        form = AttendanceForm(request.POST, instance=attendance)
        if form.is_valid():
            form.save()
            return redirect('attendance_list')
        return render(request, 'attendance/attendance_form.html', {'form': form})

class AttendanceDeleteView(View):
    def post(self, request, pk):
        attendance = get_object_or_404(AttendanceRecord, pk=pk)
        attendance.delete()
        return redirect('attendance_list')

def live_feed(request):
    return render(request, 'attendance/live_feed.html')

def is_anomalous(student):
    today = datetime.date.today()
    days_range = 2  # Number of days to check
    start_date = today - timedelta(days=days_range)

    # Generate a list of dates to check
    dates_to_check = [start_date + timedelta(days=i) for i in range(days_range)]

    # Fetch attendance records for the student within the date range
    attendance_records = AttendanceRecord.objects.filter(
        student=student,
        date__in=dates_to_check
    )

    # Extract the dates for which the student was present
    present_dates = set(record.date for record in attendance_records)

    # Determine if the student is missing from any of the dates
    missing_dates = set(dates_to_check) - present_dates

    # Return True if there are any missing dates
    return bool(missing_dates)

def overlay_emoji(frame, emoji_path, position):
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    if emoji is None:
        print("Error loading emoji image")
        return frame

    # Resize emoji if needed
    emoji = cv2.resize(emoji, (50, 50))

    # Extract alpha channel
    alpha_channel = emoji[:, :, 3]
    alpha_channel = cv2.merge((alpha_channel, alpha_channel, alpha_channel))

    # Overlay emoji on frame
    x, y = position
    h, w = emoji.shape[:2]

    # Check if emoji fits in the frame
    if x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame

    # Create a region of interest
    roi = frame[y:y+h, x:x+w]

    # Blend the emoji with the region of interest
    img1_bg = cv2.bitwise_and(roi, cv2.bitwise_not(alpha_channel))
    img2_fg = cv2.bitwise_and(emoji[:, :, :3], alpha_channel)
    dst = cv2.add(img1_bg, img2_fg)

    frame[y:y+h, x:x+w] = dst

    return frame

def send_unknown_person_notification(frame):
    # Replace with your email credentials
    sender_email = "py3140443@gmail.com"
    receiver_email = "prityadav99982@gmail.com"
    password = "ojkk tios zdgx qwme"

    # Create email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Unknown Person Detected"

    body = "An unknown person has been detected in the video feed."
    msg.attach(MIMEText(body, 'plain'))

    # Convert frame to image data
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img_data = buffer.tobytes()
    image = MIMEImage(img_data, name='unknown_person.jpg')
    msg.attach(image)

    # Send email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:  # Replace with your SMTP server
        smtp.starttls()
        smtp.login(sender_email, password)
        smtp.sendmail(sender_email, receiver_email, msg.as_string())

def generate_frames():
    os.environ['CV_CAP_MSMF'] = '0'
    camera = cv2.VideoCapture(0)
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return

    known_faces = []
    print("Loading student images...")
    for student in Student.objects.all():
        try:
            student_image_path = student.image.path
            if student_image_path:
                # Obtain embedding using VGG-Face
                embeddings = DeepFace.represent(img_path=student_image_path, model_name='VGG-Face', enforce_detection=False)
                if embeddings and 'embedding' in embeddings[0]:
                    embedding = embeddings[0]['embedding']
                    known_faces.append((embedding, student))  # Store the embedding and student object
        except Exception as e:
            print(f"Error processing student {student.name}: {str(e)}")
    
    print("Student images loaded.")
    attendance_marked = set()

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame.")
            continue  # Skip to the next iteration

        print("Extracting faces...")
        face_boxes = DeepFace.extract_faces(frame, enforce_detection=False)
        
        for box in face_boxes:
            if isinstance(box, dict):  # Ensure box is a dictionary
                x, y, w, h = box['facial_area']['x'], box['facial_area']['y'], box['facial_area']['w'], box['facial_area']['h']
            else:
                continue
            face_region = frame[y:y+h, x:x+w]
            try:
                print("1")
                face_embeddings = DeepFace.represent(img_path=face_region, model_name="VGG-Face", enforce_detection=False)
                if face_embeddings and 'embedding' in face_embeddings[0]:
                    face_embedding = face_embeddings[0]['embedding']
                    print("Face embedding obtained.")

                # Default to red color for unknown faces
                color = (0, 0, 255)
                mark = "Unknown"

                for known_image_path, student in known_faces:
                    print("2")
                    result = DeepFace.verify(face_embedding, known_image_path, model_name='VGG-Face', enforce_detection=False)
                    if result['verified']:
                        print("3")
                        if student.name not in attendance_marked:
                            attendance_marked.add(student.name)
                            print(f"Attendance marked for: {student.name}")
                            mark_attendance(student) 
                        if student.name in attendance_marked :
                            mark = student.name
                            color = (0, 255, 0)
                       
                        if is_anomalous(student):
                            frame = overlay_emoji(frame, Path(__file__).parent / 'Red-Flag.png', (x, y-60))
                        break
                    else:
                        send_unknown_person_notification(frame)

                # Draw rectangle and text on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame,f"{mark}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                print(f"Error verifying face: {str(e)}")

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def mark_attendance(student):
    AttendanceRecord.objects.get_or_create(
        student=student,
        date=datetime.date.today(),
        defaults={'time': datetime.datetime.now().time(), 'status': 'Present'}
    )
    subject = "Attendance Marked"
    message = f"Dear {student.name}, your attendance has been marked successfully."
    send_attendance_alert(student.email, subject, message)


# import os
# import logging
# import cv2
# from django.http import JsonResponse, StreamingHttpResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.shortcuts import render, redirect, get_object_or_404
# import numpy as np
# from .models import Student, AttendanceRecord
# import datetime
# from django.views import View
# from .forms import StudentForm, AttendanceForm
# from deepface import DeepFace
# from collections import defaultdict

# logger = logging.getLogger(__name__)

# known_face_encodings = []
# known_face_names = []

# def index(request):
#     return render(request, 'attendance/index.html')

# def adminn(request):
#     return render(request, 'attendance/adminn.html')

# def student_list(request):
#     students = Student.objects.all()
#     return render(request, 'attendance/student_list.html', {'students': students})

# def attendance_list(request):
#     records = AttendanceRecord.objects.all()
#     return render(request, 'attendance/attendance_list.html', {'records': records})

# def upload_camera(request):
#     return render(request, 'attendance/upload_image.html')

# @csrf_exempt
# def detect_faces(request):
#     if request.method == 'POST':
#         image_data = request.FILES.get('image')

#         if image_data is not None:
#             temp_dir = 'temp'
#             if not os.path.exists(temp_dir):
#                 os.makedirs(temp_dir)

#             image_path = os.path.join(temp_dir, image_data.name)
#             with open(image_path, 'wb') as f:
#                 f.write(image_data.read())

#             results = []
#             students = Student.objects.all()

#             for student in students:
#                 try:
#                     student_image_path = student.image.path
#                     result = DeepFace.verify(image_path, student_image_path, enforce_detection=False)

#                     if result['verified']:
#                         results.append(student.name)
#                         AttendanceRecord.objects.create(
#                             student=student,
#                             date=datetime.date.today(),
#                             time=datetime.datetime.now().time(),
#                             status='Present'
#                         )
#                 except Exception as e:
#                     logger.error(f"Error processing student {student.name}: {str(e)}")

#             return JsonResponse({'names': results})

#     return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt
def detect_faces(request):
    if request.method == 'POST':
        image_data = request.FILES.get('image')
        if image_data is not None:
            # Create temp directory if it doesn't exist
            temp_dir = 'temp'
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            image_path = os.path.join(temp_dir, image_data.name)
            with open(image_path, 'wb') as f:
                f.write(image_data.read())
            results = []
            students = Student.objects.all()
            for student in students:
                try:
                    student_image_path = student.image.path
                    result = DeepFace.verify(image_path, student_image_path, enforce_detection=False)
                    if result['verified']:
                        results.append(student.name)
                        # Mark attendance
                        AttendanceRecord.objects.create(
                            student=student,
                            date=datetime.date.today(),
                            time=datetime.datetime.now().time(),
                            status='Present'
                        )
                except Exception as e:
                    print(f"Error processing student {student.name}: {str(e)}")
            return JsonResponse({'names': results})
    return JsonResponse({'error': 'Invalid request method'}, status=405)

# class StudentListView(View):
#     def get(self, request):
#         students = Student.objects.all()
#         return render(request, 'attendance/student_list.html', {'students': students})

# class StudentCreateView(View):
#     def get(self, request):
#         form = StudentForm()
#         return render(request, 'students/student_form.html', {'form': form})

#     def post(self, request):
#         form = StudentForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             return redirect('student-list')
#         return render(request, 'students/student_form.html', {'form': form})

# class StudentUpdateView(View):
#     def get(self, request, pk):
#         student = get_object_or_404(Student, pk=pk)
#         form = StudentForm(instance=student)
#         return render(request, 'students/student_form.html', {'form': form})

#     def post(self, request, pk):
#         student = get_object_or_404(Student, pk=pk)
#         form = StudentForm(request.POST, request.FILES, instance=student)
#         if form.is_valid():
#             form.save()
#             return redirect('student-list')
#         return render(request, 'students/student_form.html', {'form': form})

# class StudentDeleteView(View):
#     def post(self, request, pk):
#         student = get_object_or_404(Student, pk=pk)
#         print(f"Deleting student: {student.name}")
#         student.delete()
#         return redirect('student-list')

# class AttendanceListView(View):
#     def get(self, request):
#         attendance_records = AttendanceRecord.objects.all()
#         return render(request, 'attendance/attendance_list.html', {'attendance_records': attendance_records})

# class AttendanceCreateView(View):
#     def get(self, request):
#         form = AttendanceForm()
#         return render(request, 'attendance/attendance_form.html', {'form': form})

#     def post(self, request):
#         form = AttendanceForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('attendance_list')
#         return render(request, 'attendance/attendance_form.html', {'form': form})

# class AttendanceEditView(View):
#     def get(self, request, pk):
#         attendance = get_object_or_404(AttendanceRecord, pk=pk)
#         form = AttendanceForm(instance=attendance)
#         return render(request, 'attendance/attendance_form.html', {'form': form})

#     def post(self, request, pk):
#         attendance = get_object_or_404(AttendanceRecord, pk=pk)
#         form = AttendanceForm(request.POST, instance=attendance)
#         if form.is_valid():
#             form.save()
#             return redirect('attendance_list')
#         return render(request, 'attendance/attendance_form.html', {'form': form})

# class AttendanceDeleteView(View):
#     def post(self, request, pk):
#         attendance = get_object_or_404(AttendanceRecord, pk=pk)
#         attendance.delete()
#         return redirect('attendance_list')

# def live_feed(request):
#     return render(request, 'attendance/live_feed.html')

# def generate_frames():
#     os.environ['CV_CAP_MSMF'] = '0'
#     camera = cv2.VideoCapture(0)

#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     if not camera.isOpened():
#         print("Error: Camera could not be opened.")
#         return

#     known_faces = []
#     students = Student.objects.all()

#     for student in students:
#         try:
#             student_image_path = student.image.path
#             if student_image_path:
#                 known_face = DeepFace.represent(img_path=student_image_path, model_name='Facenet', enforce_detection=False)
#                 known_faces.append((known_face[0]['embedding'], student))
#         except Exception as e:
#             print(f"Error processing student {student.name}: {str(e)}")

#     attendance_marked = set()

#     while True:
#         success, frame = camera.read()
#         if not success:
#             print("Error: Could not read frame.")
#             continue

#         frame = cv2.resize(frame, (640, 480))
#         faces = detect_faces(frame)

#         for (x, y, w, h) in faces:
#             face = frame[y:y+h, x:x+w]
#             verified = False
#             mark = "Unknown"  # Default to "Unknown" if not verified or in attendance

#             try:
#                 face_embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']

#                 for known_face_embedding, student in known_faces:
#                     result = DeepFace.verify(face_embedding, known_face_embedding, model_name='Facenet', enforce_detection=False)
#                     if result['verified'] and student.name not in attendance_marked:
#                         mark_attendance(student)
#                         attendance_marked.add(student.name)
#                         verified = True
#                         mark = student.name  # Update mark to student's name if verified
#                         break

#                 color = (0, 255, 0) if verified or mark in attendance_marked else (0, 0, 255)
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2) 
#                 cv2.putText(frame, mark if verified or mark in attendance_marked else mark, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             except Exception as e:
#                 print(f"Error verifying face: {str(e)}")

#         ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def train_recognizer():
#     known_faces = []
#     labels = []

#     students = Student.objects.all()
#     for student in students:
#         try:
#             student_image_path = student.image.path
#             if student_image_path:
#                 image = cv2.imread(student_image_path)
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 known_faces.append(gray)
#                 labels.append(student.id)  # Use student ID as label      
#         except Exception as e:
#             print(f"Error processing student {student.name}: {str(e)}")
#     print(labels)
#     # Train the face recognizer
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.train(known_faces, np.array(labels))
#     return recognizer

# def generate_frames(recognizer):
#     camera = cv2.VideoCapture(0)

#     if not camera.isOpened():
#         print("Error: Camera could not be opened.")
#         return

#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     attendance_marked = set()
#     students = Student.objects.all()

#     while True:
#         success, frame = camera.read()
#         if not success:
#             print("Error: Could not read frame.")
#             continue

#         frame = cv2.resize(frame, (640, 480))
#         faces = detect_faces(frame)

#         for (x, y, w, h) in faces:
#             face = frame[y:y+h, x:x+w]
#             gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#             # Initialize for this frame
#             verified_student = None
#             min_confidence = 100  # Start with a high threshold

#             # Check all students for this face
#             for student in students:
#                 label, confidence = recognizer.predict(gray_face)

#                 # Only consider if confidence is lower than the minimum found
#                 if confidence < min_confidence:
#                     min_confidence = confidence
#                     verified_student = student

#             # If a verified student is found
#             if verified_student and min_confidence < 100:
#                 if verified_student.name not in attendance_marked:
#                     print(verified_student.name)
#                     mark_attendance(verified_student)
#                     attendance_marked.add(verified_student.name)
#                 mark = verified_student.name
#                 color = (0, 255, 0)  # Green for recognized
#             else:
#                 mark = "Unknown"
#                 color = (0, 0, 255)  # Red for unknown

#             # Draw rectangle and put text on the frame
#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#             cv2.putText(frame, mark, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
# def save_recognized_frame(frame, student_name):
#     save_dir = 'recognized_faces'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = os.path.join(save_dir, f'{student_name}_{timestamp}.jpg')
#     cv2.imwrite(filename, frame)

# def video_feed(request):
#     return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

# def check_absences():
#     today = datetime.date.today()
#     start_date = today - datetime.timedelta(days=30)  # Example date range, adjust as needed
#     absence_records = AttendanceRecord.objects.filter(date__range=(start_date, today), status='Absent')

#     absence_counts = {}
#     for record in absence_records:
#         if record.student_id in absence_counts:
#             absence_counts[record.student_id] += 1
#         else:
#             absence_counts[record.student_id] = 1

#     return absence_counts

# def analyze_attendance():
#     today = datetime.date.today()
#     start_date = today - datetime.timedelta(days=30)  # Example date range, adjust as needed
#     attendance_records = AttendanceRecord.objects.filter(date__range=(start_date, today))

#     attendance_stats = {
#         'total_students': Student.objects.count(),
#         'total_days': (today - start_date).days + 1,
#         'attendance_counts': defaultdict(int),
#         'absent_counts': defaultdict(int),
#         'total_attendance': 0,
#         'total_absences': 0,
#         'attendance_percentage': {}
#     }

#     for record in attendance_records:
#         if record.status == 'Present':
#             attendance_stats['attendance_counts'][record.student_id] += 1
#             attendance_stats['total_attendance'] += 1
#         else:
#             attendance_stats['absent_counts'][record.student_id] += 1
#             attendance_stats['total_absences'] += 1

#     for student_id in attendance_stats['attendance_counts']:
#         attendance_percentage = (attendance_stats['attendance_counts'][student_id] / attendance_stats['total_days']) * 100
#         attendance_stats['attendance_percentage'][student_id] = attendance_percentage

#     return attendance_stats  
