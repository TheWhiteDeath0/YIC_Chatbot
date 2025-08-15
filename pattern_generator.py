# pattern_generator.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
m
# Load the pre-trained T5 model and tokenizer
# This might take a few minutes the first time it runs as it needs to download the model
print("Loading T5 paraphrasing model...")
try:
    model_name = 'Vamsi/T5_Paraphrase'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    print("✅ T5 Model loaded successfully.")
except Exception as e:
    print(f"[T5 Load Error] Could not load the model. Error: {e}")
    model, tokenizer = None, None

def generate_patterns(question, num_patterns=3):
    """
    Takes a question and generates a specified number of paraphrased versions.
    """
    if not all([model, tokenizer]):
        print("T5 model is not available. Skipping pattern generation.")
        return []

    # Prepare the input for the T5 model
    text = "paraphrase: " + question
    
    encoding = tokenizer.encode_plus(
        text, 
        padding='longest', 
        return_tensors="pt"
    )
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # Generate the paraphrased outputs
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,
        num_beams=10,
        num_return_sequences=num_patterns,
        early_stopping=True
    )

    # Decode the generated patterns
    generated_patterns = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != question.lower(): # Ensure it's a new pattern
            generated_patterns.append(sent)
            
    return generated_patterns

def save_new_patterns(tag, original_question, new_patterns):
    """
    Saves the newly generated patterns to a file for review.
    """
    if not new_patterns:
        return

    # Create a file to log suggested patterns
    with open("suggested_new_patterns.txt", "a") as f:
        f.write(f"--- Intent: {tag} ---\n")
        f.write(f"Original Question: {original_question}\n")
        for pattern in new_patterns:
            f.write(f"  - New Pattern: {pattern}\n")
        f.write("\n")
    print(f"✅ Saved {len(new_patterns)} new pattern suggestions to 'suggested_new_patterns.txt'")