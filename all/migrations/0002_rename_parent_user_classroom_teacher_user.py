# Generated by Django 4.2.1 on 2023-05-11 23:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('all', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='classroom',
            old_name='parent_user',
            new_name='teacher_user',
        ),
    ]
