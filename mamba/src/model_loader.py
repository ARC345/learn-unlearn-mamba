from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def load_model_and_tokenizer(model_name):
    """Load Mamba model and tokenizer."""

    # Mamba uses GPT-NeoX tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load Mamba model
    model = MambaLMHeadModel.from_pretrained(model_name)

    return model, tokenizer
