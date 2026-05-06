import os
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Setup Persistent Long-Term Memory (ChromaDB)
def get_thread_vectorstore(thread_id: str):
    """Create thread-specific vectorstore for memory isolation"""
    return Chroma(
        collection_name=f"agent_memory_thread_{thread_id}",
        embedding_function=embeddings,
        persist_directory=f"./chroma_langgraph_db/thread_{thread_id}"
    )

# Default vectorstore for thread "1"
vectorstore = get_thread_vectorstore("1")

 
# 3. Define the State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str

# 4. Node: Retrieve from Long-Term Memory
def retrieve_ltm(state: AgentState):
    # Retrieve based on the last user message
    last_message = state["messages"][-1].content
    thread_vectorstore = get_thread_vectorstore(current_thread_id)
    docs = thread_vectorstore.similarity_search(last_message, k=2)
    retrieved_content = "".join([d.page_content for d in docs])
    return {"context": retrieved_content}

# 5. Node: Call Groq Model
def call_model(state: AgentState):
    system_prompt = (
        "You are a helpful agent with access to long-term memory. "
        "Use the provided context to personalize your response."
        f"PAST CONTEXT: {state['context']}"
    )

    # Prepend the system prompt to the message history
    messages = [("system", system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def save_to_ltm(state: AgentState):
    # Save the latest user message and assistant response to long-term memory
    if len(state["messages"]) >= 2:
        last_human_msg = state["messages"][-2].content
        last_ai_msg = state["messages"][-1].content
        memory = f"User said: {last_human_msg}\nAssistant responded: {last_ai_msg}"
        thread_vectorstore = get_thread_vectorstore(current_thread_id)
        thread_vectorstore.add_texts([memory])
    return state

# Global thread ID for current session
current_thread_id = "1"

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_ltm)
workflow.add_node("model", call_model)
workflow.add_node("commit_memory", save_to_ltm)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "model")
workflow.add_edge("model", "commit_memory")
workflow.add_edge("commit_memory", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

def run_cli_chatbot(thread_id: str = "1"):
    """Interactive CLI chatbot with memory capabilities"""
    global current_thread_id
    current_thread_id = thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🤖 Memory Agent Chatbot")
    print("Type 'quit', 'exit', or press Ctrl+C to end the conversation")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye! 👋")
                break
                
            if not user_input:
                continue
                
            input_msg = HumanMessage(content=user_input)
            
            for event in app.stream({"messages": [input_msg]}, config):
                if "retrieve" in event:
                    print(f"Retrieving from memory... \n {event['retrieve']['context']}")
                if "model" in event:
                    ai_response = event['model']['messages'][-1].content
                    print(f"AI: {ai_response}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    import sys
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    run_cli_chatbot(thread_id)


