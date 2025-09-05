from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableMap
from IPython.display import Markdown, display
import logging
import os
import requests
import argparse

# =========================
# General Config
# =========================
# Set up for bypassing system proxy
os.environ["no_proxy"] = "127.0.0.1,localhost"

# Set up logging for better error handling
DISABLE_LOGGING = False  # Set to True to disable logging, False to enable
logging.basicConfig(
    level=logging.CRITICAL if DISABLE_LOGGING else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ================================
# Configure Ollama host and port
# ================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

MODEL_NAME = "llama3.2:3b"
# MODEL_NAME = "gemma3:latest" 
# MODEL_NAME = "gpt-oss:latest"

# ===========================
# Define the prompt template
# ===========================
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)
test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "code"]
)

# =========================
# Input parameters
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of 5 integers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

input_params = {"language": args.language, "task": args.task}
# input_params = {"language": "python", "task": "return a list of 5 integers"}

# =========================
# Test server connection
# =========================
def test_server_connection(url: str) -> bool:
    try:
        response = requests.get(url, timeout=5)
        logger.info(f"Ollama server status: {response.status_code}")
        return True
    except requests.RequestException as e:
        logger.error(f"❌ Failed to connect to Ollama server at {url}: {e}")
        return False

# # 1) Use Chatgpt api
# api_key = "sk-proj-Wxydnfibkzqrzu7Lw......."
# llm = OpenAI(
#     openai_api_key=api_key
# )

# 2) Use local LLM
# =================================================
# Function to invoke a model and display the result
# =================================================
def run_model(model_name, code_prompt: PromptTemplate, test_prompt: PromptTemplate, params: dict) -> None:
    try:
        logger.info(f"🚀 Running model: {model_name})")
        llm = OllamaLLM(model=model_name, base_url=OLLAMA_URL, timeout=30)


        # Step 1: Generate code
        code_chain = (
            RunnableMap({
                "task": lambda x: x["task"],
                "language": lambda x: x["language"]
            })
            | code_prompt
            | llm
            | RunnableLambda(lambda output: {
                "code": output.strip(),
                "language": input_params["language"],
                "task": input_params["task"]
            })
        )

        # Step 2: Generate test from code + language
        test_chain = (
            RunnableLambda(lambda x: {
                "code": x["code"],
                "language": x["language"]
            })
            | test_prompt
            | llm
        )

        # Step 3: Combine into sequence
        chain = RunnableSequence(
            code_chain 
            | RunnableLambda(lambda x: {
                "test": test_chain.invoke(x),
                "code": x['code']
                })
            )

        result = chain.invoke(input_params)

        print(">>>>>> GENERATED CODE:")
        print(result["code"])
        # display(Markdown(result["code"]))

        print(">>>>>> GENERATED TEST:")
        print(result["test"])
        # display(Markdown(result["test"]))

    except Exception as e:
        logger.error(f"⚠️ Error with {model_name}: {e}")


# =============================
# Main entry point
# =============================
if __name__ == '__main__':
    if not test_server_connection(OLLAMA_URL):
        exit(1)

    # Run Model
    run_model(MODEL_NAME, code_prompt, test_prompt, input_params)
