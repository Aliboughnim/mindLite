from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db.models.signals import post_save
from django.dispatch import receiver
from django_resized import ResizedImageField


    

class User(AbstractUser):
    class Role(models.TextChoices):
        PARENT = "PARENT", "Parent"
        STUDENT = "STUDENT", "Student"
        TEACHER = "TEACHER", "Teacher"
        ADMIN = "ADMIN", "Admin"

    base_role = Role.TEACHER

    role = models.CharField(max_length=50, choices=Role.choices)
    photo =  ResizedImageField(size=[120, 120   ],null=True, upload_to='user_photos/')
    
    def save(self, *args, **kwargs):
        if not self.pk:
            self.role = self.base_role
        # Add debug statement or log message
        print("Saving user:", self.username)
        # or
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Saving user: {self.username}")
        # Call the original save method
        return super().save(*args, **kwargs)


class StudentManager(BaseUserManager):
    def get_queryset(self, *args, **kwargs):
        results = super().get_queryset(*args, **kwargs)
        return results.filter(role=User.Role.STUDENT)


class Student(User):

    base_role = User.Role.STUDENT

    student = StudentManager()

    class Meta:
        proxy = True

    def welcome(self):
        return "Only for students"


@receiver(post_save, sender=Student)
def create_user_profile(sender, instance, created, **kwargs):
    if created and instance.role == "STUDENT":
        StudentProfile.objects.create(user=instance)





class TeacherManager(BaseUserManager):
    def get_queryset(self, *args, **kwargs):
        results = super().get_queryset(*args, **kwargs)
        return results.filter(role=User.Role.TEACHER)


class Teacher(User):

    base_role = User.Role.TEACHER

    teacher = TeacherManager()

    class Meta:
        proxy = True

    def welcome(self):
        return "Only for teachers"


class TeacherProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    cin = models.CharField(max_length=10)
    phone_num = models.BigIntegerField()



@receiver(post_save, sender=Teacher)
def create_user_profile(sender, instance, created, **kwargs):
    if created and instance.role == "TEACHER":
        TeacherProfile.objects.create(user=instance)

class Classroom(models.Model):
    level = models.IntegerField()
    name = models.CharField(max_length=30)
    scholar_year = models.CharField(max_length=9)
    teacher_user = models.ForeignKey(Teacher, on_delete=models.CASCADE,related_name="class_teacher")

class ParentManager(BaseUserManager):
    def get_queryset(self, *args, **kwargs):
        results = super().get_queryset(*args, **kwargs)
        return results.filter(role=User.Role.Parent)


class Parent(User):

    base_role = User.Role.PARENT

    parent = ParentManager()

    class Meta:
        proxy = True

    def welcome(self):
        return "Only for parents"
    def parent_profile(self):
        return self.parentprofile


class ParentProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE,related_name='parent_profiles')
    cin = models.CharField(max_length=10)
    phone_num = models.BigIntegerField()


@receiver(post_save, sender=Parent)
def create_user_profile(sender, instance, created, **kwargs):
    if created and instance.role == "PARENT":
        ParentProfile.objects.create(user=instance)

class StudentProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    student_id = models.IntegerField(null=True, blank=True)
    GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
    )
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES,null=True)
    parent_user = models.ForeignKey(Parent, on_delete=models.CASCADE,related_name="children")
    enrolled_class = models.ForeignKey(Classroom, on_delete=models.CASCADE)


class Course(models.Model):
    title= models.TextField(null=True)
    text = models.TextField()
    summary = models.TextField()
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE,related_name="teacher_courses")
    classroom = models.ForeignKey(Classroom, on_delete=models.CASCADE,related_name="class_courses")
    audio = models.TextField(null=True)
    image = models.ImageField(upload_to='img/course_images', null=True)

class Problem(models.Model):
    text = models.TextField()
    date = models.DateField(auto_now_add=True)
    answer = models.TextField()
    graph = models.TextField(null=True)
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE,related_name="teacher_problems")
    classroom = models.ForeignKey(Classroom, on_delete=models.CASCADE,related_name="class_problems")




class Xp(models.Model):
    xp= models.IntegerField()
    date = models.DateTimeField(auto_now_add=True)
    Student = models.ForeignKey(Student, on_delete=models.CASCADE)


class Chat(models.Model):
    sender = models.ForeignKey(User, related_name='chat_sender', on_delete=models.CASCADE)
    receiver = models.ForeignKey(User, related_name='chat_receiver', on_delete=models.CASCADE)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)


class Done(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    problem = models.ForeignKey(Problem, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    result = models.BooleanField()  

class Quiz(models.Model):
    question = models.TextField()
    answer = models.TextField()
    distractor1 = models.TextField()
    distractor2 = models.TextField()
    distractor3 = models.TextField()
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='quizzes',null=True)