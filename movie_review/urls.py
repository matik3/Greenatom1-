from django.contrib import admin
from django.urls import path
from reviews import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.review_view, name='review'),
]

