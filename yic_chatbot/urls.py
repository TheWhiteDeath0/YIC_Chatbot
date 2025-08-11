# yic_chatbot/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Change 'api/' to '' to include chatbot URLs from the root
    path('', include('chatbot.urls')), 
]