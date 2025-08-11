# chatbot/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # This rule handles the homepage
    path('', views.chat_home, name='chat_home'),
    
    # This rule handles the chat API logic
    path('api/chat/', views.chat_api, name='chat_api'),
]