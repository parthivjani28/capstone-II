# forms.py
from django import forms
from .models import Student,AttendanceRecord
from django.contrib.auth.forms import AuthenticationForm

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ['name', 'image', 'birthday', 'department','email','phone_number', 'semester', 'gender']
        widgets = {
            'birthday': forms.DateInput(attrs={'type': 'date'}),
            'gender': forms.Select(), 
        }

class AttendanceForm(forms.ModelForm):
    class Meta:
        model = AttendanceRecord
        fields = ['student', 'date', 'time', 'status']
        widgets = {
            'student': forms.Select(attrs={'class': 'student'}),
            'date': forms.DateInput(attrs={'type': 'date', 'class': 'date'}),
            'time': forms.TimeInput(attrs={'type': 'time', 'class': 'time'}),
            'status': forms.Select(attrs={'class': 'status'}),
        }
        
class CustomLoginForm(AuthenticationForm):
    username = forms.CharField(label='Username', max_length=254)
    password = forms.CharField(label='Password', widget=forms.PasswordInput)