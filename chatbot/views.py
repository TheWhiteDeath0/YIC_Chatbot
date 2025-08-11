# chatbot/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .nlp_processor import get_response

# This function shows the HTML page
def chat_home(request):
    return render(request, 'index.html')

# This is your API function
@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            # get_response now returns a dictionary
            bot_response_data = get_response(user_message)
            
            # Return the entire dictionary as a JSON response
            return JsonResponse(bot_response_data)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)