"""
evaluation.py

Loads the fine-tuned LoRA adapter and performs
Sanskrit -> English translation.

Author: Sohil Bhimani
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------------------------
# Configuration
# -------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "../checkpoints"   # Change if your adapter is stored elsewhere

# -------------------------------------------------
# Load Tokenizer
# -------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# -------------------------------------------------
# Load Base Model
# -------------------------------------------------

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# -------------------------------------------------
# Load LoRA Adapter
# -------------------------------------------------

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Model loaded successfully!\n")


# -------------------------------------------------
# Translation Function
# -------------------------------------------------

def translate(sanskrit_sentence):

    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Translate the following Sanskrit sentence into English.

Sanskrit:
{sanskrit_sentence}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Extract only assistant response
    if "assistant" in decoded.lower():
        decoded = decoded.split("assistant")[-1]

    return decoded.strip()


# -------------------------------------------------
# Main Program
# -------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" Sanskrit → English Translator ")
    print("=" * 60)

    while True:

        sentence = input("\nEnter Sanskrit sentence (or type 'exit'): ")

        if sentence.lower() == "exit":
            break

        translation = translate(sentence)

        print("\nEnglish Translation:\n")
        print(translation)
        print("\n" + "-" * 60)




