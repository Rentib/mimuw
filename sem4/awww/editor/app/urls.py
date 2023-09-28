from django.urls import path

from . import views

app_name = 'editor'

urlpatterns = [
    path('', views.index, name='index'),
    path('load_tree/', views.load_tree, name='load_tree'),
    path('load_file/', views.load_file, name='load_file'),
    path('add_child/', views.add_child, name='add_child'),
    path('delete_item/', views.delete_item, name='delete_item'),
    path('select_file/', views.select_file, name='select_file'),
    path('save_file/', views.save_file, name='save_file'),
    path('compile/', views.compile, name='compile'),
]
