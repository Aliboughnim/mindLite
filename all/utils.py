from django.shortcuts import get_object_or_404
from .models import * 
from django.db import connection

def get_teachers_for_parent(id):

    with connection.cursor() as cursor:
        cursor.execute("SELECT enrolled_class_id FROM all_studentprofile WHERE parent_user_id= %s",[id])
        classes = cursor.fetchall()
        

   