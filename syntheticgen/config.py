import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    # Model configurations
    LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "models/openhermes-2.5-mistral-7b.Q4_K_M.gguf")
    
    # OpenRouter configurations (optional)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL = os.getenv("MODEL_NAME", "anthropic/claude-2")  # Default model
    
    # Path configurations
    SCHEMA_DIR = os.path.join(os.path.dirname(__file__), 'schema')
    EXAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'example_data')

    # MODEL = os.getenv("MODEL_NAME", "huggingfaceh4/zephyr-7b-beta")
    # MODEL = os.getenv("MODEL_NAME", "mistralai/mistral-7b-instruct")
    # MODEL = os.getenv("MODEL_NAME", "google/gemini-2.5-pro-exp-03-25")

    CHUNK_SIZE = 500  # Records per API call
    MAX_RETRIES = 3
    DELAY_BETWEEN_CALLS = 1  # Seconds