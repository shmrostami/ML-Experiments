from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
import os

# Setup memory
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history"
)

# Setup prompt with a placeholder for the chat history
prompt = ChatPromptTemplate(
    input_variables=["input"],
    messages=[
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

# Configure Ollama host and port
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Initialize Ollama LLM
chat = OllamaLLM(model="llama3.2:3b", base_url=OLLAMA_URL, timeout=30)

# Create a runnable chain
chain = {"input": RunnablePassthrough()} | prompt | chat

# Wrap the chain with message history
conversational_chain = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="response"
)

# Session management (simple session ID for this example)
session_id = "default_session"

while True:
    content = input(">> ")
    if content.lower() == 'bye':
        break
    result = conversational_chain.invoke(
        {"input": content},
        config={"configurable": {"session_id": session_id}}
    )

    # Handle potential key changes in output
    if "response" in result:
        print(result["response"])
    else:
        print(result.get("text", "No response available"))  # Fallback for different output keys