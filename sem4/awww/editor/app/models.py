from django.db import models
from django.contrib.auth.models import User
from datetime import date

# Create your models here.

class Directory(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    available = models.BooleanField(default=True)
    availability_changed_at = models.DateTimeField(auto_now=True)
    last_content_change_at = models.DateTimeField(auto_now=True)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)

    def print(self):
        if not self.available: return ""
        result = "<ul>"
        result += f"<li class='folder' data-id='{self.id}'>"
        result += f"{self.name} " if self.parent else ""
        result += "<a class='add-file-btn'><i class='fas fa-file-circle-plus'></i></a>"
        result += "<a class='add-dir-btn'><i class='fas fa-folder-plus'></i></a>"
        result += "<a class='delete-btn'><i class='fas fa-trash'></i></a></li>" if self.parent else "</li>"
        dirs = self.directory_set.all()
        files = self.file_set.all()

        if dirs or files:
            for d in dirs:
                result += d.print()
            for f in files:
                result += f.print()

        result += "</ul>"
        return result

    def delete(self):
        self.available = False
        self.availability_changed_at = date.today()
        self.save()
        dirs = self.directory_set.all()
        files = self.file_set.all()
        for d in dirs:
            d.delete()
        for f in files:
            f.delete()

    def __str__(self):
        return self.name

    @staticmethod
    def find_root(user):
        root = Directory.objects.filter(parent=None, owner=user).first()
        if not root:
            root = Directory(name='root', owner=user)
            root.save()
        return root

class File(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    available = models.BooleanField(default=True)
    availability_changed_at = models.DateTimeField(auto_now=True)
    last_content_change_at = models.DateTimeField(auto_now=True)
    parent = models.ForeignKey(Directory, on_delete=models.CASCADE)
    content = models.TextField(blank=True)

    def print(self):
        if not self.available: return ""
        result = "<ul>"
        result += f"<li data-id='{self.id}'><a class='file-select-btn'>{self.name}</a>"
        result += "<a class='delete-btn'><i class='fas fa-trash'></i></a></li>"
        result += "</ul>"
        return result

    def delete(self):
        self.available = False
        self.availability_changed_at = date.today()
        self.save()

    def __str__(self):
        return self.name


class Section(models.Model):
    name = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    start_line = models.IntegerField()
    end_line = models.IntegerField()
    start_char = models.IntegerField(null=True, blank=True)
    end_char = models.IntegerField(null=True, blank=True)
    section_type = models.CharField(max_length=255)
    section_category = models.CharField(max_length=255)
    section_status = models.CharField(max_length=255)
    status_data = models.TextField(blank=True)
    content = models.TextField()
    # TODO: implement sections (actually just write some chat-gpt query)
