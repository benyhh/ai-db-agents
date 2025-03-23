import tiktoken
import chromadb
from openai import OpenAI
import os

def num_tokens_from_message(message, model="gpt-3.5-turbo"):
    """Return the number of tokens used by a message."""
    encoding = tiktoken.encoding_for_model(model)
    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
        
        num_tokens = tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
        return num_tokens
    else:
        # For other models, just encode the message content
        return len(encoding.encode(message["content"]))

def trim_messages_by_tokens(messages, system_message, max_tokens):
    """Trim messages to stay under the token limit while preserving recent context."""
    # Always include system message in token count
    system_tokens = num_tokens_from_message(system_message)
    available_tokens = max_tokens - system_tokens
    
    # Start from most recent messages, add until we hit the token limit
    trimmed_messages = []
    token_count = 0
    
    for message in reversed(messages):  # Process from newest to oldest
        message_tokens = num_tokens_from_message(message)
        if token_count + message_tokens <= available_tokens:
            trimmed_messages.insert(0, message)  # Add to front of list
            token_count += message_tokens
        else:
            break  # Stop if adding another message would exceed the limit
    
    return trimmed_messages

def get_embedding(text, client, model="text-embedding-3-small"):
    """Get embedding for a text using OpenAI's embedding API."""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def get_chroma_client(persist_directory=None):
    """Initialize and return a ChromaDB client."""
    if persist_directory:
        return chromadb.PersistentClient(path=persist_directory)
    else:
        return chromadb.Client()

def generate_search_query(current_prompt, message_history, openai_client):
    """
    Generate a context-aware search query based on the current prompt and conversation history.
    
    Args:
        current_prompt: The user's current question
        message_history: List of previous messages
        openai_client: OpenAI client for generating the query
        
    Returns:
        An expanded query that incorporates relevant context
    """
    if not message_history:
        return current_prompt
    
    # Format the conversation history
    formatted_history = []
    
    for msg in message_history[-4:]:  # Take at most the last 2 exchanges (4 messages)
        formatted_history.append(f"{msg['role'].upper()}: {msg['content']}")
    
    history_text = "\n".join(formatted_history)
    
    system_prompt = """
    Based on the conversation history and the user's current question, generate a detailed search query 
    that would help retrieve relevant information. The query should be self-contained and include all 
    necessary context from the conversation history. Focus on specific entities, names, and details mentioned.
    
    Your response should be ONLY the search query, nothing else - no explanations or formatting.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONVERSATION HISTORY:\n{history_text}\n\nCURRENT QUESTION: {current_prompt}\n\nGENERATE SEARCH QUERY:"}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=100
        )
        
        expanded_query = response.choices[0].message.content.strip()
        return expanded_query
    except Exception as e:
        print(f"Error generating search query: {e}")
        return current_prompt  # Fallback to the original prompt

def get_relevant_documents(query, client, openai_client, collection_name, top_k=3, embedding_model="text-embedding-3-small"):
    """
    Get relevant documents from ChromaDB collection based on query similarity.
    
    Args:
        query: User's query text
        client: ChromaDB client
        openai_client: OpenAI client for generating embeddings
        collection_name: Name of the collection to search
        top_k: Number of documents to retrieve
        embedding_model: Model to use for embeddings
        
    Returns:
        List of document contents
    """
    try:
        # First generate an embedding for the query
        query_embedding = get_embedding(query, openai_client, model=embedding_model)
        
        collection = client.get_collection(name=collection_name)
        
        # Search using the embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Extract document contents
        if results and 'documents' in results and len(results['documents']) > 0:
            return results['documents'][0]  # First element contains the list of documents
        
        return []
    except Exception as e:
        print(f"Error retrieving documents from ChromaDB: {e}")
        return []

def format_context(documents):
    """Format retrieved documents into context for the LLM with strict instructions."""
    if not documents:
        return ""
    
    formatted = """ATTENTION: Your response must be based EXCLUSIVELY on the following retrieved information. DO NOT use any knowledge outside of these documents:

"""
    for i, doc in enumerate(documents, 1):
        formatted += f"Document {i}:\n{doc}\n\n"
    
    formatted += "Remember: Only use the information provided above. If you cannot find the answer in these documents, state that clearly."
    
    return formatted
