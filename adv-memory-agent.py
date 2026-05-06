import os
import sys
import json
import shutil
import gc
import time
import atexit
from typing import Annotated, TypedDict, Optional
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

load_dotenv()

# =======================
# CONFIGURATION
# =======================
DB_PATH = "./chroma_langgraph_db"
CHECKPOINTS_PATH = "./checkpoints"
THREADS_METADATA_PATH = "./threads_metadata.json"

# Ensure directories exist
Path(DB_PATH).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINTS_PATH).mkdir(parents=True, exist_ok=True)

# =======================
# GLOBAL STATE & CLEANUP
# =======================
_vectorstore_cache = {}
_thread_cleanup_handlers = {}

def cleanup_all_connections():
    """Cleanup all database connections on exit"""
    print("\n🧹 Cleaning up connections...")
    
    # Close all vectorstore connections
    for thread_id, vs in _vectorstore_cache.items():
        try:
            if hasattr(vs, '_client'):
                vs._client.close()
        except:
            pass
    
    _vectorstore_cache.clear()
    gc.collect()
    print("✅ Cleanup complete")

# Register cleanup handler
atexit.register(cleanup_all_connections)

# =======================
# LLM & EMBEDDINGS SETUP
# =======================
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# =======================
# VECTORSTORE MANAGEMENT
# =======================
def get_thread_vectorstore(thread_id: str):
    """
    Get or create thread-specific vectorstore with caching.
    Reuses connections to avoid duplicate database access.
    """
    if thread_id in _vectorstore_cache:
        return _vectorstore_cache[thread_id]
    
    collection_name = f"agent_memory_thread_{thread_id}"
    persist_dir = f"{DB_PATH}/thread_{thread_id}"
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        _vectorstore_cache[thread_id] = vectorstore
        return vectorstore
    except Exception as e:
        print(f"⚠️ Error creating vectorstore for thread {thread_id}: {e}")
        raise

def close_thread_vectorstore(thread_id: str):
    """Properly close vectorstore connection for a thread"""
    if thread_id not in _vectorstore_cache:
        return
    
    try:
        vs = _vectorstore_cache[thread_id]
        if hasattr(vs, '_client'):
            vs._client.close()
    except Exception as e:
        print(f"⚠️ Error closing vectorstore: {e}")
    finally:
        _vectorstore_cache.pop(thread_id, None)
        gc.collect()
        time.sleep(0.2)  # Allow OS to release file locks

# =======================
# STATE DEFINITION
# =======================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str

# =======================
# WORKFLOW NODES
# =======================
def retrieve_ltm(state: AgentState):
    """Retrieve relevant context from long-term memory"""
    try:
        last_message = state["messages"][-1].content
        vectorstore = get_thread_vectorstore(get_current_thread_id())
        
        # Similarity search with error handling
        docs = vectorstore.similarity_search(last_message, k=2)
        retrieved_content = "\n".join([d.page_content for d in docs]) if docs else ""
        
        return {"context": retrieved_content}
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        return {"context": ""}

def call_model(state: AgentState):
    """Call the LLM with context from long-term memory"""
    try:
        system_prompt = (
            "You are a helpful agent with access to long-term memory. "
            "Use the provided context to personalize your response. "
            "Be concise and helpful.\n"
        )
        
        if state["context"]:
            system_prompt += f"\n📚 RELEVANT PAST CONTEXT:\n{state['context']}\n"
        
        messages = [("system", system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        print(f"⚠️ Model error: {e}")
        return {"messages": [HumanMessage(content=f"Error: {str(e)}")]}

def save_to_ltm(state: AgentState):
    """Save the conversation turn to long-term memory"""
    try:
        if len(state["messages"]) >= 2:
            last_human_idx = -2
            last_ai_idx = -1
            
            # Find actual human and AI messages (skip system)
            messages = state["messages"]
            if len(messages) >= 2:
                last_human_msg = messages[last_human_idx].content
                last_ai_msg = messages[last_ai_idx].content
                
                memory_entry = (
                    f"[USER]: {last_human_msg}\n"
                    f"[ASSISTANT]: {last_ai_msg}"
                )
                
                vectorstore = get_thread_vectorstore(get_current_thread_id())
                vectorstore.add_texts([memory_entry])
        
        return state
    except Exception as e:
        print(f"⚠️ Memory save error: {e}")
        return state

# =======================
# THREAD MANAGEMENT
# =======================
_current_thread_id = "1"

def get_current_thread_id():
    """Get the current thread ID"""
    global _current_thread_id
    return _current_thread_id

def set_current_thread_id(thread_id: str):
    """Set the current thread ID"""
    global _current_thread_id
    _current_thread_id = thread_id

def load_threads_metadata():
    """Load metadata about all threads"""
    if os.path.exists(THREADS_METADATA_PATH):
        try:
            with open(THREADS_METADATA_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_threads_metadata(metadata):
    """Save thread metadata"""
    with open(THREADS_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

def get_available_threads():
    """Get list of all available threads"""
    metadata = load_threads_metadata()
    return list(metadata.keys()) if metadata else []

def thread_exists(thread_id: str):
    """Check if a thread exists"""
    metadata = load_threads_metadata()
    return thread_id in metadata

def create_thread_metadata(thread_id: str, description: str = ""):
    """Create metadata for a new thread"""
    metadata = load_threads_metadata()
    
    if thread_id not in metadata:
        metadata[thread_id] = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_accessed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "message_count": 0
        }
        save_threads_metadata(metadata)
        print(f"✅ Created thread: {thread_id}")
    else:
        print(f"⚠️ Thread {thread_id} already exists")
    
    return metadata.get(thread_id)

def update_thread_metadata(thread_id: str, message_count: int = None):
    """Update thread metadata"""
    metadata = load_threads_metadata()
    
    if thread_id in metadata:
        metadata[thread_id]["last_accessed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if message_count is not None:
            metadata[thread_id]["message_count"] = message_count
        save_threads_metadata(metadata)

# =======================
# WORKFLOW SETUP
# =======================
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_ltm)
workflow.add_node("model", call_model)
workflow.add_node("commit_memory", save_to_ltm)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "model")
workflow.add_edge("model", "commit_memory")
workflow.add_edge("commit_memory", END)

# Setup persistent checkpointing with SQLite
checkpoint_dir = CHECKPOINTS_PATH
checkpointer = SqliteSaver(checkpoint_dir)
app = workflow.compile(checkpointer=checkpointer)

# =======================
# MEMORY OPERATIONS
# =======================
def reset_thread_memory(thread_id: str, force: bool = False):
    """
    Reset memory for a specific thread.
    Properly closes connections before deletion.
    """
    # Step 1: Close vectorstore connection
    close_thread_vectorstore(thread_id)
    time.sleep(0.1)
    
    # Step 2: Delete thread directory
    thread_dir = f"{DB_PATH}/thread_{thread_id}"
    if os.path.exists(thread_dir):
        try:
            shutil.rmtree(thread_dir)
            print(f"✅ Reset memory for thread: {thread_id}")
            return True
        except PermissionError as e:
            print(f"❌ Permission denied. Retrying with force close...")
            if force:
                try:
                    # Try alternative approach
                    for root, dirs, files in os.walk(thread_dir, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(thread_dir)
                    print(f"✅ Force reset memory for thread: {thread_id}")
                    return True
                except Exception as e2:
                    print(f"❌ Error: {e2}")
                    print("💡 Try closing all Python instances and running again")
                    return False
        except Exception as e:
            print(f"❌ Error deleting thread {thread_id}: {e}")
            return False
    else:
        print(f"⚠️ No memory found for thread: {thread_id}")
        return False

def reset_all_memories(force: bool = False):
    """Reset memories for all threads"""
    # Close all connections
    for thread_id in list(_vectorstore_cache.keys()):
        close_thread_vectorstore(thread_id)
    
    time.sleep(0.2)
    
    # Delete all data
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"✅ Reset all thread memories")
            Path(DB_PATH).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    else:
        print(f"⚠️ No memory data found")
        return False

def show_thread_memory_usage(thread_id: str):
    """Show memory usage for a thread"""
    thread_dir = f"{DB_PATH}/thread_{thread_id}"
    if os.path.exists(thread_dir):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(thread_dir)
            for filename in filenames
        )
        size_kb = total_size / 1024
        print(f"💾 Thread '{thread_id}' memory: {size_kb:.2f} KB")
    else:
        print(f"⚠️ No memory data for thread: {thread_id}")

# =======================
# INTERACTIVE CLI
# =======================
def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print("🤖 MEMORY AGENT CHATBOT WITH PERSISTENT CHECKPOINTING")
    print("="*60)
    print("\n📋 COMMANDS:")
    print("  /list          - List all threads")
    print("  /new <id>      - Create new thread (e.g., /new work)")
    print("  /switch <id>   - Switch to thread (e.g., /switch work)")
    print("  /info          - Show current thread info")
    print("  /memory        - Show memory usage of current thread")
    print("  /history       - Show checkpoints for current thread")
    print("  /reset         - Clear current thread memory")
    print("  /reset-all     - Clear all memories (careful!)")
    print("  /quit or /exit - Exit chatbot")
    print("="*60 + "\n")

def print_thread_info(thread_id: str):
    """Print information about a thread"""
    metadata = load_threads_metadata()
    if thread_id in metadata:
        info = metadata[thread_id]
        print(f"\n📌 Thread: {thread_id}")
        print(f"   Created: {info['created_at']}")
        print(f"   Last accessed: {info['last_accessed']}")
        print(f"   Messages: {info['message_count']}")
        if info.get('description'):
            print(f"   Description: {info['description']}")
    else:
        print(f"⚠️ Thread '{thread_id}' has no metadata")

def list_all_threads():
    """List all available threads"""
    metadata = load_threads_metadata()
    if not metadata:
        print("📭 No threads found. Create one with: /new <thread_id>")
        return
    
    print("\n📚 AVAILABLE THREADS:")
    print("-" * 50)
    for thread_id, info in metadata.items():
        marker = "👉" if thread_id == get_current_thread_id() else "  "
        desc = f" - {info['description']}" if info.get('description') else ""
        print(f"{marker} {thread_id:15} | Messages: {info['message_count']:3}{desc}")
    print("-" * 50 + "\n")

def show_checkpoints(thread_id: str):
    """Show checkpoint history for a thread"""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoints = list(app.get_state_history(config))
    
    if not checkpoints:
        print(f"📭 No checkpoints for thread: {thread_id}")
        return
    
    print(f"\n🔖 CHECKPOINTS FOR THREAD '{thread_id}':")
    print("-" * 60)
    for i, checkpoint in enumerate(reversed(checkpoints[-5:]), 1):  # Show last 5
        step = checkpoint.metadata.get('step', 'N/A')
        messages = checkpoint.values.get('messages', [])
        if messages:
            last_msg = messages[-1].content
            truncated = (last_msg[:50] + "...") if len(last_msg) > 50 else last_msg
            print(f"{i}. Step {step}: {truncated}")
    print("-" * 60 + "\n")

def run_interactive_cli():
    """Main interactive CLI loop"""
    print_welcome()
    
    # Create default thread if none exists
    if not get_available_threads():
        create_thread_metadata("default", "Default conversation thread")
        set_current_thread_id("default")
    else:
        # Use first available thread
        set_current_thread_id(get_available_threads()[0])
    
    thread_id = get_current_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    message_count = 0
    
    print(f"💬 Current thread: {thread_id}")
    print("Type a message or /help for commands\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # ==================
            # COMMAND HANDLING
            # ==================
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()
                args = user_input.split()[1:] if len(user_input.split()) > 1 else []
                
                if cmd in ["/quit", "/exit"]:
                    print("\n👋 Goodbye! Your memories are saved.")
                    break
                
                elif cmd == "/help":
                    print_welcome()
                
                elif cmd == "/list":
                    list_all_threads()
                
                elif cmd == "/new":
                    if not args:
                        print("⚠️ Usage: /new <thread_id> [description]")
                    else:
                        new_thread_id = args[0]
                        description = " ".join(args[1:]) if len(args) > 1 else ""
                        create_thread_metadata(new_thread_id, description)
                
                elif cmd == "/switch":
                    if not args:
                        print("⚠️ Usage: /switch <thread_id>")
                    else:
                        new_thread_id = args[0]
                        if thread_exists(new_thread_id):
                            set_current_thread_id(new_thread_id)
                            thread_id = new_thread_id
                            config = {"configurable": {"thread_id": thread_id}}
                            message_count = load_threads_metadata().get(thread_id, {}).get('message_count', 0)
                            print(f"✅ Switched to thread: {thread_id}")
                        else:
                            print(f"❌ Thread '{new_thread_id}' not found")
                            list_all_threads()
                
                elif cmd == "/info":
                    print_thread_info(thread_id)
                
                elif cmd == "/memory":
                    show_thread_memory_usage(thread_id)
                
                elif cmd == "/history":
                    show_checkpoints(thread_id)
                
                elif cmd == "/reset":
                    confirm = input("⚠️ Clear memory for this thread? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        reset_thread_memory(thread_id, force=True)
                
                elif cmd == "/reset-all":
                    confirm = input("⚠️⚠️ Clear ALL thread memories? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        reset_all_memories(force=True)
                
                else:
                    print("❓ Unknown command. Type /help for available commands")
                
                continue
            
            # ==================
            # CHAT MODE
            # ==================
            input_msg = HumanMessage(content=user_input)
            
            # Stream the graph execution
            retrieved_context = None
            ai_response = None
            
            for event in app.stream({"messages": [input_msg]}, config):
                if "retrieve" in event:
                    retrieved_context = event["retrieve"].get("context", "")
                    if retrieved_context:
                        print(f"\n💭 Context from memory: {retrieved_context[:100]}...")
                
                if "model" in event:
                    ai_response = event["model"]["messages"][-1].content
                    print(f"\n🤖 Assistant: {ai_response}\n")
            
            # Update metadata
            message_count += 1
            update_thread_metadata(thread_id, message_count)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Your memories are saved.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    try:
        run_interactive_cli()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_all_connections()