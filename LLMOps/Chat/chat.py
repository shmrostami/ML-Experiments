from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationChain
import os

memory = ConversationBufferMemory( return_messages=True)

# Setup prompt with a placeholder for the chat history
prompt = ChatPromptTemplate(
    input_variables=["input"],
    messages=[
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# ================================
# Configure Ollama host and port
# ================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

chat = OllamaLLM(model="llama3.2:3b", base_url=OLLAMA_URL, timeout=30)

chain = ConversationChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    input_key="input",
    output_key="response"
)

while True:
    content = input(">> ")
    if content.lower() == 'bye':
        break
    
    result = chain.invoke({"input": content})

    print(result["response"])