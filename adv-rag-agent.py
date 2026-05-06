"""Advanced RAG agent with memory, tools, and conditional routing."""

import json
import os
from typing import Annotated, TypedDict, Literal
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from vector_db import VectorDB

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

vdb = VectorDB()
search = DuckDuckGoSearchRun()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_thread_vectorstore(thread_id: str):
    """Create thread-specific vectorstore for memory isolation"""
    return Chroma(
        collection_name=f"agent_memory_thread_{thread_id}",
        embedding_function=embeddings,
        persist_directory=f"./chroma_langgraph_db/thread_{thread_id}"
    )

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. 
    Supports basic arithmetic: +, -, *, /, **, parentheses.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def text_analyzer(text: str) -> str:
    """Analyze text and provide statistics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Statistics including word count, character count, and sentence count
    """
    words = text.split()
    sentences = text.split('.') + text.split('!') + text.split('?')
    sentence_count = len([s for s in sentences if len(s.strip()) > 0])
    
    stats = {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(text.replace(' ', '')),
        "sentence_count": sentence_count,
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
    }
    
    return json.dumps(stats, indent=2)

tools = [search, calculator, text_analyzer]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    use_rag: bool
    rag_ctx: list[str]
    thread_id: str

def rag_decider(state: AgentState):
    """Decide whether to use RAG based on the current message."""
    last_message = state["messages"][-1].content
    
    sys = """You are a RAG decision agent. Analyze the user's message and determine if it would benefit from 
    retrieval-augmented generation using a knowledge base containing information about NLP, GANs, and Transformers.
    
    Consider using RAG if:
    - The message involves technical explanations about NLP, deep learning, or AI concepts
    - The user asks for definitions, explanations, or comparisons
    - The topic is related to machine learning architectures or techniques
    - The query requires factual knowledge about AI/ML concepts
    
    Respond ONLY with a JSON object: {"use_rag": true/false, "reason": "brief explanation"}
    """
    
    msgs = [
        SystemMessage(content=sys),
        HumanMessage(content=last_message)
    ]
    
    resp = llm.invoke(msgs).content.strip()
    resp = resp.replace("```json", "").replace("```", "").strip()
    
    try:
        dec = json.loads(resp)
        use_rag = dec.get("use_rag", False)
    except json.JSONDecodeError:
        use_rag = False
    
    print(f"[RAG Decider] Use RAG: {use_rag}")
    
    return {"use_rag": use_rag, "rag_ctx": []}

def rag_retriever(state: AgentState):
    """Retrieve relevant documents using RAG."""
    if not state["use_rag"]:
        return state
    
    print("[RAG Retriever] Retrieving relevant documents...")
    
    last_message = state["messages"][-1].content
    queries = [last_message]
    
    rag_ctx = []
    seen_docs = set()
    
    for q in queries:
        try:
            results = vdb.search(q, k=2)
            for res in results:
                doc_text = res['doc']
                if doc_text not in seen_docs:
                    rag_ctx.append(doc_text)
                    seen_docs.add(doc_text)
        except Exception as e:
            print(f"Error searching for query '{q}': {e}")
            continue
    
    print(f"[RAG Retriever] Retrieved {len(rag_ctx)} relevant documents")
    
    return {"rag_ctx": rag_ctx}

def retrieve_ltm(state: AgentState):
    """Retrieve from long-term memory based on thread_id."""
    thread_id = state.get("thread_id", "1")
    last_message = state["messages"][-1].content
    thread_vectorstore = get_thread_vectorstore(thread_id)
    docs = thread_vectorstore.similarity_search(last_message, k=2)
    retrieved_content = "".join([d.page_content for d in docs])
    print(f"[Memory] Retrieved context for thread {thread_id}")
    return {"context": retrieved_content}

def call_model(state: AgentState):
    """Call the LLM with tools, memory context, and RAG context."""
    system_prompt = (
        "You are a helpful AI agent with access to tools, memory, and a knowledge base. "
        "Use tools ONLY when explicitly needed:\n"
        "- calculator: ONLY when user asks for mathematical calculations\n"
        "- text_analyzer: ONLY when user asks to analyze text statistics\n"
        "- duckduckgo_search: ONLY when user asks for current web information\n"
        "For general questions, especially those answered by RAG context or memory, respond directly WITHOUT using tools. "
        "Use the provided context from memory and RAG to personalize your response."
    )
    
    # Build context string
    context_parts = []
    if state.get("context"):
        context_parts.append(f"PAST MEMORY CONTEXT:\n{state['context']}")
    if state.get("rag_ctx"):
        context_parts.append(f"RAG KNOWLEDGE BASE:\n" + "\n".join([f"• {ctx}" for ctx in state["rag_ctx"]]))
    
    if context_parts:
        system_prompt += "\n\n" + "\n\n".join(context_parts)
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    print(f"[Model] Generated response, tool_calls: {len(response.tool_calls) if response.tool_calls else 0}")
    
    return {"messages": [response]}

def save_to_ltm(state: AgentState):
    """Save conversation to long-term memory."""
    thread_id = state.get("thread_id", "1")
    
    if len(state["messages"]) >= 2:
        last_human_msg = state["messages"][-2].content
        last_ai_msg = state["messages"][-1].content
        memory = f"User said: {last_human_msg}\nAssistant responded: {last_ai_msg}"
        thread_vectorstore = get_thread_vectorstore(thread_id)
        thread_vectorstore.add_texts([memory])
        print(f"[Memory] Saved context for thread {thread_id}")
    
    return state

workflow = StateGraph(AgentState)

workflow.add_node("rag_decider", rag_decider)
workflow.add_node("rag_retriever", rag_retriever)
workflow.add_node("retrieve", retrieve_ltm)
workflow.add_node("model", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("commit_memory", save_to_ltm)

workflow.add_edge(START, "rag_decider")
workflow.add_edge("rag_decider", "rag_retriever")
workflow.add_edge("rag_retriever", "retrieve")
workflow.add_edge("retrieve", "model")

# Conditional routing: if model calls tools, go to tools, otherwise to memory
workflow.add_conditional_edges(
    "model",
    tools_condition,
    {
        "tools": "tools",
        END: "commit_memory"
    }
)

workflow.add_edge("tools", "model")
workflow.add_edge("commit_memory", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

def run_cli_chatbot(thread_id: str = "1"):
    """Interactive CLI chatbot with memory, tools, and RAG."""
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🤖 Advanced RAG Agent with Memory & Tools")
    print("Features: RAG knowledge base, Long-term memory, Web search, Calculator, Text analyzer")
    print("Type 'quit', 'exit', or press Ctrl+C to end the conversation")
    print("-" * 70)
    
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
                if "rag_decider" in event:
                    print(f"  [RAG Decider] Deciding whether to use knowledge base...")
                if "rag_retriever" in event:
                    print(f"  [RAG Retriever] Searching knowledge base...")
                if "retrieve" in event:
                    print(f"  [Memory] Retrieving from long-term memory...")
                if "model" in event:
                    model_output = event['model']['messages'][-1]
                    if model_output.tool_calls:
                        print(f"  [Tools] Calling {len(model_output.tool_calls)} tool(s)...")
                    else:
                        ai_response = model_output.content
                        print(f"AI: {ai_response}")
                if "tools" in event:
                    for tool_msg in event['tools']['messages']:
                        if isinstance(tool_msg, ToolMessage):
                            print(f"  [Tools] Tool result: {tool_msg.content[:100]}...")
                if "commit_memory" in event:
                    print(f"  [Memory] Saved to long-term memory")
                    
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    import sys
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    run_cli_chatbot(thread_id)
