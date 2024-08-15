from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
   path('', views.index, name='index'), 
   path("adminn/",views.adminn,name='adminn'),
    path('students/', views.StudentListView.as_view(), name='student-list'),
    path('students/add/', views.StudentCreateView.as_view(), name='student-add'),
    path('students/edit/<int:pk>/', views.StudentUpdateView.as_view(), name='student-edit'),
      path('students/delete/<int:pk>/', views.StudentDeleteView.as_view(), name='student-delete'),
      path('attendance/', views.AttendanceListView.as_view(), name='attendance_list'),
    path('attendance/add/', views.AttendanceCreateView.as_view(), name='attendance-add'),
    path('attendance/edit/<int:pk>/',views.AttendanceEditView.as_view(), name='attendance-edit'),
    path('attendance/delete/<int:pk>/', views.AttendanceDeleteView.as_view(), name='attendance-delete'),

    path('upload/' ,views.upload_camera,name ="upload"),
   path('detect_faces/', views.detect_faces, name='detect_faces'),
    path('live_feed/', views.live_feed, name='live_feed'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('login/', views.loginn, name='login'),
    path('logout/', views.logout_view, name='logout'),
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)