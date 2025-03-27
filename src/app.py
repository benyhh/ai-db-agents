import streamlit as st
from openai import OpenAI
import os
import json
import sqlite3
from dotenv import load_dotenv
import tiktoken
import pandas as pd
from utils import (
    num_tokens_from_message, 
    trim_messages_by_tokens, 
    get_chroma_client,
    get_relevant_documents,
    format_context,
    generate_search_query
)
# SQL imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.agent_toolkits import create_sql_agent
# Import for direct SQL query approach
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pyprojroot import here


# Load environment variables
load_dotenv(here(".env"))

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize LangChain LLM for SQL agent
langchain_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initialize ChromaDB client
chroma_persist_dir = os.path.join(here(), "data", "db", "chroma")
chroma_client = get_chroma_client(chroma_persist_dir)
COLLECTION_NAME = "imdb"  # Update with your collection name

# Initialize SQL Database for SQL Agent
sqldb_path = os.path.join(here(), "data", "db", "sql", "imdb.db")
sql_db = SQLDatabase.from_uri(f"sqlite:///{sqldb_path}")

# Initialize SQL Agent
@st.cache_resource
def get_sql_agent():
    return create_sql_agent(
        langchain_llm, 
        db=sql_db, 
        agent_type="openai-tools", 
        verbose=True
    )

# Initialize Direct SQL Query chain
@st.cache_resource
def get_sql_chain():
    # Create a chain that writes SQL queries
    write_query_chain = create_sql_query_chain(langchain_llm, sql_db)
    # agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
    
    # Create a tool that executes SQL queries
    execute_query_tool = QuerySQLDataBaseTool(db=sql_db)
    
    return write_query_chain, execute_query_tool

# Set page configuration
st.set_page_config(page_title="AI Database Agent Chat", page_icon="💬", layout="wide")

# Constants
MAX_TOKENS = 4000  # Maximum number of tokens to keep in context (configurable)

# Initialize session state for chat history and settings
if "messages" not in st.session_state:
    st.session_state.messages = []
if "use_rag" not in st.session_state:
    st.session_state.use_rag = True
if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "query_method" not in st.session_state:
    st.session_state.query_method = "rag" # Default to RAG
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []
if "sql_approach" not in st.session_state:
    st.session_state.sql_approach = "direct" # "agent" or "direct"

# Function to execute SQL query directly with SQLite
def execute_sql_query(query):
    try:
        conn = sqlite3.connect(sqldb_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return str(e)

# Function to generate response from OpenAI with RAG support
def generate_response(prompt):
    context = ""
    debug_entry = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": prompt,
        "method": st.session_state.query_method,
        "steps": []
    }
    
    # Choose query method based on user selection
    if st.session_state.query_method == "rag" and st.session_state.use_rag:
        # RAG approach
        try:
            # Generate a context-aware search query
            search_query = generate_search_query(
                prompt,
                st.session_state.messages,
                client
            )
            
            debug_entry["steps"].append({
                "step": "Generate search query",
                "input": prompt,
                "output": search_query
            })
            
            # Store for debugging
            st.session_state.search_query = search_query
            
            # Use the enhanced query for document retrieval
            documents = get_relevant_documents(
                search_query, 
                chroma_client,
                client,
                COLLECTION_NAME, 
                st.session_state.top_k
            )
            
            debug_entry["steps"].append({
                "step": "Vector search",
                "query": search_query,
                "results": documents
            })
            
            context = format_context(documents)
            st.session_state.last_retrieved_docs = documents
            st.session_state.last_method = "RAG"
        except Exception as e:
            error_msg = f"Error retrieving context: {e}"
            st.error(error_msg)
            debug_entry["error"] = error_msg
    
    elif st.session_state.query_method == "sql":
        # Choose SQL approach based on user selection
        if st.session_state.sql_approach == "direct":
            # Direct SQL approach - clearer and more visible queries
            try:
                # Get the SQL query chain and execution tool
                write_query_chain, execute_query_tool = get_sql_chain()
                
                # Generate the SQL query
                with st.spinner("Generating SQL query..."):
                    sql_query = write_query_chain.invoke({"question": prompt})
                
                # Display the SQL query (for debugging)
                with st.container():
                    st.write("### Generated SQL Query:")
                    st.code(sql_query, language="sql")
                
                # Execute the SQL query
                with st.spinner("Executing SQL query..."):
                    # First, store the query and result in pandas for display
                    result_df = execute_sql_query(sql_query)
                    if isinstance(result_df, pd.DataFrame):
                        st.write("### SQL Query Result:")
                        st.dataframe(result_df)
                        # Convert to string for the LLM response
                        sql_result = result_df.to_string()
                    else:
                        st.error(f"Error executing SQL: {result_df}")
                        sql_result = f"Error: {result_df}"
                
                # Add to debug info
                debug_entry["steps"].append({
                    "step": "SQL Query Generation",
                    "query": sql_query
                })
                
                debug_entry["steps"].append({
                    "step": "SQL Query Execution",
                    "result": sql_result,
                    "dataframe": result_df if isinstance(result_df, pd.DataFrame) else None
                })
                
                # Generate a natural language response using the LLM with stricter constraints
                answer_prompt = PromptTemplate.from_template(
                    """You are answering a question EXCLUSIVELY based on the provided SQL query results.
                    
                    Question: {question}
                    SQL Query: {query}
                    SQL Result: {result}
                    
                    IMPORTANT INSTRUCTIONS:
                    1. ONLY use the information from the SQL result to answer the question.
                    2. Do NOT use any external knowledge that is not in the provided SQL result.
                    3. If the SQL result does not contain enough information to fully answer the question, state this clearly.
                    4. Do not make up or infer information that is not explicitly present in the results.
                    
                    Answer: """
                )
                
                # Generate final answer
                with st.spinner("Generating answer..."):
                    messages = [
                        {"role": "system", "content": answer_prompt.format(
                            question=prompt, 
                            query=sql_query, 
                            result=sql_result
                        )},
                    ]
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.7,
                    )
                    
                    answer = response.choices[0].message.content
                
                # Record method and return
                debug_entry["steps"].append({
                    "step": "Final Answer Generation",
                    "answer": answer
                })
                
                st.session_state.last_method = "SQL Direct"
                st.session_state.sql_query = sql_query
                st.session_state.sql_result = sql_result
                
                debug_entry["response"] = answer
                st.session_state.debug_info.append(debug_entry)
                return answer
                
            except Exception as e:
                error_msg = f"Error using Direct SQL approach: {str(e)}"
                st.error(error_msg)
                debug_entry["error"] = error_msg
                st.session_state.debug_info.append(debug_entry)
                return error_msg
                
        else:
            # SQL Agent approach
            try:
                # Get the SQL agent
                sql_agent = get_sql_agent()
                
                # Create a placeholder for intermediate steps if enabled
                intermediate_steps_container = None
                if st.session_state.get("show_intermediate_steps", True):
                    intermediate_steps_container = st.container()
                    with intermediate_steps_container:
                        st.write("### Agent execution steps:")
                
                # Execute the SQL agent with more explicit tracking
                with st.spinner("Executing SQL Agent..."):
                    with get_openai_callback() as cb:
                        if intermediate_steps_container and st.session_state.get("show_intermediate_steps", True):
                            st_callback = StreamlitCallbackHandler(intermediate_steps_container)
                            agent_response = sql_agent.invoke(
                                {"input": prompt},
                                {"callbacks": [st_callback]}
                            )
                        else:
                            agent_response = sql_agent.invoke({"input": prompt})
                        
                        # Store API usage stats
                        debug_entry["openai_usage"] = {
                            "total_tokens": cb.total_tokens,
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_cost": cb.total_cost,
                        }
                
                # Store the entire agent response
                debug_entry["full_agent_response"] = str(agent_response)
                
                # Process intermediate steps
                if 'intermediate_steps' in agent_response:
                    for i, step in enumerate(agent_response['intermediate_steps']):
                        debug_entry["steps"].append({
                            "step": f"Agent Step {i+1}",
                            "content": str(step)
                        })
                        
                        # Try to extract SQL queries directly from the step
                        if len(step) >= 2:
                            # Usually, step[0] is action, step[1] is observation
                            action = step[0]
                            observation = step[1]
                            
                            # Try to extract SQL query
                            if hasattr(action, 'tool') and action.tool == 'sql_db_query':
                                # Extract and store SQL query
                                if hasattr(action, 'tool_input'):
                                    if isinstance(action.tool_input, dict) and 'query' in action.tool_input:
                                        sql_query = action.tool_input['query']
                                        debug_entry["steps"].append({
                                            "step": f"SQL Query #{i+1}",
                                            "query": sql_query,
                                            "result": str(observation)
                                        })
                                        
                                        # Execute the SQL query to ensure we see the results
                                        result_df = execute_sql_query(sql_query)
                                        if isinstance(result_df, pd.DataFrame):
                                            st.write(f"### SQL Query {i+1} Results:")
                                            st.code(sql_query, language="sql")
                                            st.dataframe(result_df)
                
                # Return the agent's response with wrapper
                answer = "Based solely on the SQL database query results, " + agent_response['output']
                debug_entry["response"] = answer
                st.session_state.debug_info.append(debug_entry)
                return answer
                
            except Exception as e:
                error_msg = f"Error using SQL Agent: {str(e)}"
                st.error(error_msg)
                debug_entry["error"] = error_msg
                st.session_state.debug_info.append(debug_entry)
                return error_msg
    
    # Prepare messages for API call
    # First trim the message history to fit within token limit
    trimmed_history = trim_messages_by_tokens(st.session_state.messages, system_message, MAX_TOKENS)
    
    messages = [
        system_message,
        *trimmed_history
    ]
    
    # Add context from RAG or SQL as a system message if available
    if context:
        context_message = {"role": "system", "content": context + "\n\nIMPORTANT: Base your response SOLELY on the information provided above. Do NOT use any other knowledge."}
        messages.append(context_message)
    
    # Add the user's current prompt - DON'T add it again if it's already the last message
    if not trimmed_history or trimmed_history[-1]["role"] != "user" or trimmed_history[-1]["content"] != prompt:
        messages.append({"role": "user", "content": prompt})
    
    debug_entry["steps"].append({
        "step": "Final LLM call",
        "messages": [{"role": m["role"], "content": m["content"][:200] + ("..." if len(m["content"]) > 200 else "")} 
                    for m in messages]  # Truncate long messages for debug display
    })
    
    # Calculate total tokens for debugging/display
    total_tokens = sum(num_tokens_from_message(msg) for msg in messages)
    st.session_state.current_token_count = total_tokens
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )
        
        answer = response.choices[0].message.content
        debug_entry["response"] = answer
        # Store the debug entry
        st.session_state.debug_info.append(debug_entry)
        return answer
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        debug_entry["error"] = error_msg
        # Store the debug entry even if there's an error
        st.session_state.debug_info.append(debug_entry)
        return error_msg

# Create system message
system_message = {"role": "system", "content": "You are an assistant that answers questions about movies and actors. IMPORTANT: You must ONLY use the information that is explicitly provided in the context and not rely on any internal knowledge. If the necessary information is not in the provided context, say that you don't have sufficient information to answer accurately."}

# Create tabs for main chat and debug view
main_tab, debug_tab = st.tabs(["Chat", "Debug View"])

# Main chat tab
with main_tab:
    st.title("AI Database Agent Chat")
    st.subheader("Ask questions about movies and actors")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for user question
    if prompt := st.chat_input("Ask a question about movies or actors"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {st.session_state.query_method.upper()}..."):
                response = generate_response(prompt)
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Debug view tab
with debug_tab:
    st.title("Debug Information")
    
    if not st.session_state.debug_info:
        st.info("No debug information available yet. Ask a question in the Chat tab to see details here.")
    else:
        # Display a selector for which query to view
        debug_entries = st.session_state.debug_info
        timestamps = [f"{i+1}. {entry['timestamp']} - {entry['method'].upper()}: {entry['query'][:50]}..." 
                     for i, entry in enumerate(debug_entries)]
        
        selected_entry_idx = st.selectbox("Select interaction to view:", 
                                         options=range(len(timestamps)),
                                         format_func=lambda i: timestamps[i])
        
        if selected_entry_idx is not None:
            entry = debug_entries[selected_entry_idx]
            
            st.subheader("Query Details")
            st.write(f"**Timestamp:** {entry['timestamp']}")
            st.write(f"**Method:** {entry['method'].upper()}")
            st.write(f"**User Query:** {entry['query']}")
            
            if "error" in entry:
                st.error(f"Error: {entry['error']}")
            
            # Special section for SQL queries
            if entry['method'] == 'sql':
                # Find SQL query steps
                sql_steps = [step for step in entry.get("steps", []) 
                            if step.get("step", "").startswith("SQL Query")]
                
                if sql_steps:
                    st.subheader("SQL Queries and Results")
                    for i, step in enumerate(sql_steps):
                        with st.expander(f"SQL Query {i+1}", expanded=i==0):
                            if "query" in step:
                                st.code(step["query"], language="sql")
                            if "result" in step:
                                st.subheader("Result")
                                st.text_area(f"Raw Result", step["result"], height=150)
                                
                                # Execute button
                                if st.button(f"Execute Query {i+1} Now", key=f"exec_{selected_entry_idx}_{i}"):
                                    try:
                                        import sqlite3
                                        conn = sqlite3.connect(sqldb_path)
                                        results_df = pd.read_sql_query(step["query"], conn)
                                        conn.close()
                                        st.subheader("Results as Table")
                                        st.dataframe(results_df)
                                    except Exception as e:
                                        st.error(f"Error executing query: {e}")
                else:
                    st.info("No SQL queries were extracted from this interaction.")
                    
                    # Show full agent response for debugging
                    if "full_agent_response" in entry:
                        with st.expander("Full Agent Response (for debugging)"):
                            st.text(entry["full_agent_response"])
            
            st.subheader("Execution Steps")
            for i, step in enumerate(entry.get("steps", [])):
                # Skip SQL Query steps as they're already displayed above for SQL method
                if entry['method'] == 'sql' and step.get("step", "").startswith("SQL Query"):
                    continue
                    
                with st.expander(f"Step {i+1}: {step['step']}", expanded=i==0):
                    for key, value in step.items():
                        if key != "step" and key != "query" and key != "result":  # Skip what we've already shown for SQL queries
                            if isinstance(value, list):
                                if key == "messages":
                                    st.write(f"**{key.capitalize()}:**")
                                    for msg in value:
                                        st.text(f"{msg['role']}: {msg['content']}")
                                else:
                                    st.write(f"**{key.capitalize()}:**")
                                    for item in value:
                                        st.write(f"- {str(item)}")
                            elif isinstance(value, dict):
                                st.write(f"**{key.capitalize()}:**")
                                st.json(value)
                            else:
                                st.write(f"**{key.capitalize()}:** {value}")
            
            if "response" in entry:
                st.subheader("Final Response")
                st.write(entry["response"])

# Sidebar settings
with st.sidebar:
    st.title("Settings")
    
    # Query method selection
    st.subheader("Query Method")
    query_method = st.radio(
        "Select how to retrieve information:",
        ["RAG", "SQL Agent"],
        index=0 if st.session_state.query_method == "rag" else 1
    )
    st.session_state.query_method = query_method.lower().replace(" ", "_")
    
    # Show appropriate settings based on method
    if st.session_state.query_method == "rag":
        # RAG settings
        st.subheader("RAG Settings")
        st.session_state.use_rag = st.checkbox("Enable RAG", value=st.session_state.use_rag)
        
        if st.session_state.use_rag:
            st.session_state.top_k = st.slider(
                "Number of documents to retrieve", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.top_k
            )
    else:
        # SQL Agent settings
        st.subheader("SQL Agent Settings")
        
        # Select SQL approach
        sql_approach = st.radio(
            "SQL Execution Method:",
            ["Direct SQL", "LangChain Agent"],
            index=0 if st.session_state.sql_approach == "direct" else 1,
            help="Direct SQL: Clearer queries and results. LangChain Agent: More powerful but less transparent."
        )
        st.session_state.sql_approach = "direct" if sql_approach == "Direct SQL" else "agent"
        
        # Show intermediate steps for agent
        if st.session_state.sql_approach == "agent":
            st.session_state.show_intermediate_steps = st.checkbox(
                "Show intermediate steps", 
                value=st.session_state.get("show_intermediate_steps", True),
                help="Display the agent's thinking process in real-time"
            )
    
    # Token settings
    st.subheader("Token Settings")
    new_token_limit = st.slider(
        "Max Context Tokens", 
        min_value=1000, 
        max_value=16000, 
        value=MAX_TOKENS, 
        step=500,
        help="Maximum token count for conversation context"
    )
    
    # Update token limit if changed
    if new_token_limit != MAX_TOKENS:
        MAX_TOKENS = new_token_limit
        st.session_state.token_limit = new_token_limit
    
    st.markdown("---")
    st.title("About")
    st.markdown("""
    This chatbot uses OpenAI's API to answer questions about movies and actors.
    
    **Example questions:**
    - What is the lowest rated movie with more than 1 million votes?
    - How many movies has Tom Hanks played in?
    - Name two actors that have played together in more than 4 movies.
    - Name two popular movies about chess.
    """)
    
    # Display token usage info
    current_tokens = sum(num_tokens_from_message(msg) for msg in st.session_state.messages) + num_tokens_from_message(system_message)
    st.info(f"Current context: {current_tokens} tokens (Max: {MAX_TOKENS})")
    
    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Clear Debug Info"):
            st.session_state.debug_info = []
            st.rerun()
