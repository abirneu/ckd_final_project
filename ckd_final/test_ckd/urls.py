from django.urls import path
from . import views

urlpatterns = [
   
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('predict/', views.predict, name='predict'),
    path('doctor_dashboard/', views.doctor_dashboard, name='doctor_dashboard'),
    path('patient_info/', views.patient_info_id, name='patient_info'),
    path('patient/<str:phone>/', views.patient_profile, name='patient_profile'),
    # path('logout/', views.logout_view, name='logout'),
    path('logout/', views.logout_view, name='logout'),
    



]
