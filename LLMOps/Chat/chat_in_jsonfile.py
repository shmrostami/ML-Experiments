# Chat with saving history in Json
import os
from typing import Dict
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Global Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
HISTORY_DIR = "chat_history"

def setup_environment() -> None:
    """Ensure local Ollama traffic bypasses proxies and create history directory."""
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    os.environ.setdefault("HTTPX_NO_PROXY", "127.0.0.1,localhost")
    os.makedirs(HISTORY_DIR, exist_ok=True)  # Create history directory if it doesn't exist

def create_chat_chain(model_name: str) -> tuple[RunnableWithMessageHistory, Dict]:
    """
    Create a conversational chain with persistent file-based history.

    Args:
        model_name: Name of the Ollama model to use.

    Returns:
        chain_with_history: RunnableWithMessageHistory instance
        history_store: Dict holding chat histories by session_id
    """
    try:
        # Initialize Ollama LLM
        llm = OllamaLLM(model=model_name, base_url=OLLAMA_URL, timeout=30)

        # Define prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful AI assistant. "
             "Respond in Persian if the user speaks Persian, otherwise in English."),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{content}")
        ])

        # Base chain (prompt → model)
        chain = prompt | llm

        # Session-based history storage
        history_store: Dict[str, FileChatMessageHistory] = {}

        def get_session_history(session_id: str) -> FileChatMessageHistory:
            """Retrieve or create chat history for the session."""
            history_file = os.path.join(HISTORY_DIR, f"chat_history_{session_id}.json")
            if session_id not in history_store:
                history_store[session_id] = FileChatMessageHistory(history_file)
            return history_store[session_id]

        # Add history management
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="content",
            history_messages_key="messages"
        )

        return chain_with_history, history_store

    except Exception as e:
        raise

def main():
    """Run an interactive chat with persistent file-based history."""
    setup_environment()

    model_name = "llama3.2:3b"

    # Create chat chain and history store
    try:
        chain_with_history, history_store = create_chat_chain(model_name)
    except Exception as e:
        return

    print("Start chatting (type 'exit' to quit):")
    session_id = "default_session"

    while True:
        user_input = input(">> ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # Run the chain
            response = chain_with_history.invoke(
                {"content": user_input, "messages": []},
                config={"configurable": {"session_id": session_id}}
            )

            print(response.strip())

            # Debug: show current history (optional, can be removed)
            # print("History:", history_store[session_id].messages)

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()