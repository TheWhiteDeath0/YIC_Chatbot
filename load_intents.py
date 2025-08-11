# load_intents.py

import json
from chatbot.models import Intent, Pattern, Response
import os

def run():
    # --- Clear all old data from the database ---
    print("Clearing old database entries...")
    Intent.objects.all().delete()
    Pattern.objects.all().delete()
    Response.objects.all().delete()
    print("Old entries cleared.")

    # --- Load the new data from intents.json ---
    # Get the base directory of the project
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # The path to your intents.json file
    json_path = os.path.join(BASE_DIR, 'intents.json')

    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # --- Loop through the JSON data and create database entries ---
    for intent_data in data['intents']:
        # Create the Intent object
        intent = Intent.objects.create(tag=intent_data['tag'])

        # Create all Pattern objects for this intent
        for pattern_text in intent_data['patterns']:
            Pattern.objects.create(intent=intent, text=pattern_text)
        
        # Create all Response objects for this intent
        for response_text in intent_data['responses']:
            Response.objects.create(intent=intent, text=response_text)

    print("✅ All intents, patterns, and responses have been successfully loaded into the database from intents.json!")

# This makes the script runnable
if __name__ == '__main__':
    run()