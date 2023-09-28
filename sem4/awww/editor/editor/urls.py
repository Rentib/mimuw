"""
URL configuration for editor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

import app.views as views

urlpatterns = [
    # path('', include('app.urls')), # BUG: breaks reverse
    path('', views.index, name='index'),
    path('app/', views.app, name='app'),
    path('load_tree/', views.load_tree, name='load_tree'),
    path('load_file/', views.load_file, name='load_file'),
    path('add_child/', views.add_child, name='add_child'),
    path('delete_item/', views.delete_item, name='delete_item'),
    path('select_file/', views.select_file, name='select_file'),
    path('save_file/', views.save_file, name='save_file'),
    path('compile/', views.compile, name='compile'),
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
]
