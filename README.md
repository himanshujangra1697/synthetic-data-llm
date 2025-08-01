# Synthetic Data Generator

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT/Apache2.0-green.svg)

A multi-engine synthetic data generation tool developed by [Himanshu Jangra](https://github.com/himanshujangra1697) for creating domain-specific datasets using cutting-edge AI models.

## Features -

- **Three Generation Engines**:
  - Local LLM (OpenHermes)
  - Cloud API (OpenRouter)
  - Statistical Model (CTGAN)
  
- **Customizable** via `config.py`:
  - Model configurations
  - Token limits
  - Batch processing controls
  - Schema validation

- **User-Friendly Interface**:
  - Streamlit web app
  - JSON schema upload
  - Real-time preview

## Generation Engines

### 1. OpenHermes (Local LLM)
**Model:** OpenHermes-2.5-Mistral-7B (Quantized)<br>
**Provider:** HuggingFace<br>
**Location:** Local Machine<br>
**Best For:** Small datasets (1-1000 rows)<br>

**Input Requirements:** Needs No. of Rows, Schema JSON

**Advantages:**
- Fully offline operation
- No API costs
- Optimized for structured JSON output

### 2. OpenRouter (Cloud API)
**Model:** mistralai/mistral-7b-instruct<br>
**Provider:** OpenRouter.ai<br>
**Location:** Cloud<br>
**Best For:** Medium datasets (100-10,000 rows)<br>

**Input Requirements:** Needs No. of Rows, Schema JSON

**Advantages:**
- Handles complex schemas
- Faster than local LLM
- Supports larger batches

**Configuration:**
```python
# config.py
OPENROUTER_API_KEY = "your_key"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"  # Can switch to claude-2, GPT-4, Llama2, etc.
MAX_TOKENS = 4000
```

### 3. CTGAN + Faker (Statistical)
**Model:** Conditional Tabular GAN<br>
**Provider:** SDV + Faker<br>
**Location:** Cloud<br>
**Best For:** Large datasets (10,000+ rows)<br>

**Input Requirements:** No. of Rows, Schema JSON, Example rows>=2 in JSON (Optional)

**Unique Capabilities:**
- Learns from example data
- Preserves statistical properties
- Supports conditional generation

**Configuration:**
No Configuration required.

## Installation

```bash
# 1. Clone repository
git clone https://github.com/himanshujangra1697/synthetic-data-llm.git
cd synthetic-data-llm

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (optional)
wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf -P models/
```

## Usage

```bash
streamlit run app.py
```

### Workflow:

1. Select the generation engine
2. Upload JSON schema
3. Set row count
4. Generate & download data

## Example Schemas

See `schema/` directory for:<br>
`patient_schema.json` - Healthcare data (For all engines)<br>
`patient_schema_example.json` - Healthcare data (Only for CTGAN)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/[YourFeatureName]`)
3. Commit your changes (`git commit -m 'Add some [YourFeatureName]'`)
4. Push to the branch (`git push origin feature/[YourFeatureName]`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.<br><br><br>

---

Developed with ❤️ by Himanshu Jangra | Data Science Enthusiast
