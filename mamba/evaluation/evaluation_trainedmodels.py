import os
import argparse
import pandas as pd
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch

def generate_text(model, tokenizer, prompt, device, max_new_tokens=100):
    eos_token_id = tokenizer.eos_token_id

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_new_tokens,
            cg=True,
            eos_token_id=eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    generated_text = generated_text.replace(prompt, "").strip()

    stop_tokens = ['</s>', '[/INST]', '[INST]']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]

    if '   Output:' in generated_text:
        generated_text = generated_text.split('   Output:')[0]

    unwanted_tokens = ['<s>', '[INST]', '[/INST]', '</s>']
    for token in unwanted_tokens:
        generated_text = generated_text.replace(token, '')

    if generated_text.startswith(' '):
        generated_text = generated_text[1:]

    return generated_text


def generate_responses(models_dir, model_prefix=None, test_csv=None, output_dir=None):
    """Generate responses for all models matching prefix.

    Args:
        models_dir: Directory containing model subdirectories.
        model_prefix: If set, only evaluate models whose name starts with this prefix.
        test_csv: Path to test CSV with 'Question' column.
        output_dir: Where to write output CSVs. Defaults to cwd.

    Returns:
        List of (model_name, output_csv_path) tuples.
    """
    if test_csv is None:
        test_csv = os.path.join(os.path.dirname(__file__), '../../data/test/D_test.csv')
    if output_dir is None:
        output_dir = '.'

    base_df = pd.read_csv(test_csv)
    text_column = 'Question'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for model_name in sorted(os.listdir(models_dir)):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        if model_prefix and not model_name.startswith(model_prefix):
            continue

        print(f"Generating responses for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = MambaLMHeadModel.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        responses = []
        for text in base_df[text_column]:
            prompt = f"<s> [INST] {text} [/INST] Output:"
            generated_response = generate_text(model, tokenizer, prompt, device, max_new_tokens=100)
            print(generated_response)
            responses.append(generated_response)

        response_df = pd.DataFrame({text_column: base_df[text_column], 'Answer': responses})
        output_csv_path = os.path.join(output_dir, f'{model_name}.csv')
        response_df.to_csv(output_csv_path, index=False)
        results.append((model_name, output_csv_path))

        del model
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate responses from trained Mamba models")
    parser.add_argument("--model-prefix", type=str, default=None,
                        help="Only evaluate models whose name starts with this prefix (e.g. 'mamba-1.4b')")
    parser.add_argument("--models-dir", type=str, default="../results",
                        help="Directory containing model subdirectories")
    parser.add_argument("--test-csv", type=str, default=None,
                        help="Path to test CSV (default: ../../data/test/D_test.csv)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to write output CSVs")
    args = parser.parse_args()

    generate_responses(
        models_dir=args.models_dir,
        model_prefix=args.model_prefix,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
