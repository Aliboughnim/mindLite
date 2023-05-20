from django.contrib import admin
from django.urls import path, include
from . import views 

urlpatterns = [
    path('',views.home,name="home"),
    path('login',views.my_login,name="login"),
    path('p_home',views.p_home,name="p_home"),
    path('t_home',views.t_home,name="t_home"),
    path('t_dash',views.t_dashboard,name="t_dash"),
    path('s_home',views.s_home,name="s_home"),
    path('s_todo',views.s_todo,name="s_todo"),
    path('contact',views.contact,name="contact"),
    path('chat',views.chat,name="chat"),
    path('t_class/<int:id>/', views.t_class, name='t_class'),
    path('t_courses/<int:classroom_id>/', views.t_courses, name='t_courses'),
    path('t_chat/<int:id>/', views.t_chat, name='t_chat'),
    path('p_chat/<int:id>/', views.p_chat, name='p_chat'),
    path('teacher_create_index/<int:classroom_id>/', views.teacher_create_index, name='teacher_create_index'),
    path('create_course/<int:classroom_id>/', views.create_course, name='create_course'),
    path("teacher_courses_details/<int:course_id>/<int:classroom_id>/", views.teacher_courses_details, name="teacher_courses_details"),
    path('t_problems/<int:classroom_id>/', views.t_problems, name='t_problems'),
    path('t_createP/<int:classroom_id>/', views.t_createP, name='t_createP'),
    path('t_createP2/<int:classroom_id>/', views.t_createP2, name='t_createP2'),
    path('graph/<int:classroom_id>/', views.graph, name='graph'),
    path('process_input/<int:classroom_id>/', views.process_input, name='process_input'),
    path('help/<int:problem_id>/',views.helpView,name='helpView'),
    path('save_message/', views.save_message, name='save_message'),
    path('save_message_p/', views.save_message_p, name='save_message_p'),
    path('upload_photo/', views.upload_photo, name='upload_photo'),
    path('upload_photo_p/', views.upload_photo_p, name='upload_photo_p'),
    path('upload_photo_s/', views.upload_photo_s, name='upload_photo_s'),
    path('logout', views.my_logout, name='logout')
]