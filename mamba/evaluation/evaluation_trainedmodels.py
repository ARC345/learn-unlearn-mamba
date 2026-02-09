import os
import pandas as pd
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch

# Directory containing all the models
models_directory = '../results'  # contains mamba-130m-finetuned and mamba-130m-charflip

def generate_text(model, tokenizer, prompt, device, max_new_tokens=100):
    # Get the token ID for the eos token
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # Generate text
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            cg=True,
            eos_token_id=eos_token_id,
        )

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # Remove prompt and return the generated text
    generated_text = generated_text.replace(prompt, "").strip()

    # Truncate at first EOS or stop token (Mamba doesn't auto-stop)
    stop_tokens = ['</s>', '[/INST]', '[INST]']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]

    # Truncate at "Output:" to avoid repetition loop
    if '   Output:' in generated_text:
        generated_text = generated_text.split('   Output:')[0]

    unwanted_tokens = ['<s>', '[INST]', '[/INST]', '</s>']
    for token in unwanted_tokens:
        generated_text = generated_text.replace(token, '')

    # Check if there's an extra space at the start
    if generated_text.startswith(' '):
        generated_text = generated_text[1:]

    return generated_text

# Load CSV
csv_file_path = '../../data/test/D_test.csv'
base_df = pd.read_csv(csv_file_path)
text_column = 'Question'

# Iterate over each model in the directory
for model_name in os.listdir(models_directory):
    model_path = os.path.join(models_directory, model_name)
    if os.path.isdir(model_path):  # Ensure it's a directory
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = MambaLMHeadModel.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        print(model_name)

        responses = []
        for text in base_df[text_column]:
            prompt = f"<s> [INST] {text} [/INST] Output:"
            generated_response = generate_text(model, tokenizer, prompt, device, max_new_tokens=100)

            print(generated_response)

            responses.append(generated_response)

        response_df = pd.DataFrame({text_column: base_df[text_column], 'Answer': responses})

        output_csv_path = f'{model_name}.csv'  # Dynamic file name
        response_df.to_csv(output_csv_path, index=False)

        del model
        torch.cuda.empty_cache()
