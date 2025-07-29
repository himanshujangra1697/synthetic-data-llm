# from syntheticgen.main import generate_with_llama

# data = generate_with_llama("syntheticgen/schema/patient_schema.json", num_rows=2)
# print(data)


import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
        model_id
        )

prompt = """
    Generate exactly 3 synthetic JSON records matching this schema:
    {
        "Name": "string",
        "Age": "int:18-90",
        "Condition": "category:[Diabetes, Hypertension, Asthma]",
        "Last_Visit": "date:past",
        "Blood_Pressure": "string:pattern=^\\d{2,3}/\\d{2,3}$"
    }

    Rules:
    1. Use only the field names and types specified.
    2. For categories, pick ONLY from the provided options.
    3. For dates, use YYYY-MM-DD.
    4. Output MUST be a valid JSON array with no extra text.

    Example
    {
        "Name": "John Doe",
        "Age": 45,
        "Condition": "Hypertension",
        "Last_Visit": "2023-08-15",
        "Blood_Pressure": "120/80"
    }

    Return ONLY a JSON array. No additional text. No Explanation.
    """

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Input:", inputs)

outputs = model.generate(
        **inputs,
        max_new_tokens=4000,
        temperature=0.7,
        do_sample=True,
        eos_token_id=None,  # Try removing EOS constraint
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:", response)

# Extract JSON array from response (Llama may add extra text)
start_idx = response.find("[")
end_idx = response.rfind("]") + 1

try:
    json_ouput = json.loads(response[start_idx:end_idx])
    print("JSON Output:", json_ouput)
except json.JSONDecodeError:
    raise ValueError("Llama returned invalid JSON. Try a simpler prompt.")

# print("JSON: ", json.dumps(json_data))