from django.db import models

class Student(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    SEMESTER_CHOICES = [(str(i), f"Semester {i}") for i in range(1, 9)]

    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='student_images/')
    registered_at = models.DateTimeField(auto_now_add=True)
    email = models.EmailField(unique=True, null=True, blank=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    birthday = models.DateField(null=True, blank=True)
    department = models.CharField(max_length=100,default='BCA')
    semester = models.CharField(max_length=2, choices=SEMESTER_CHOICES, default='1')
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='O')
    
    def __str__(self):
        return self.name
    
    def full_details(self):
        return (f"Name: {self.name}, Email: {self.email}, Phone: {self.phone_number}, "
                f"Address: {self.address}, Birthday: {self.birthday}, Department: {self.department}, "
                f"Semester: {self.semester}, Gender: {self.gender}")

class AttendanceRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    status = models.CharField(max_length=10, choices=[('Present', 'Present'), ('Absent', 'Absent')])
    
    def __str__(self):
        return f"{self.student.name} - {self.date} - {self.status}"
    
    

