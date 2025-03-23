# Testing
This folder contains notebooks for testing connections to language models and databases within the project.

## Files
* **01_llm_connection.ipynb**: Tests the connection to OpenAI's API and demonstrates basic interactions with language models.

* **02_sql_agent.ipynb**: Demonstrates querying the IMDB sample database using both direct SQL queries and SQL agents with LangChain.

* **03_embedding_and_search.ipynb**: Shows how to add documents to a vector database (ChromaDB) and perform similarity searches.

* **04_rag_agent.ipynb**: Implements a Retrieval Augmented Generation (RAG) agent that uses document retrieval to answer questions.

## Usage
These notebooks are intended for testing your setup and experimenting with different components before implementing them in the main application. Make sure you have set up your .env file with your OpenAI API key and other necessary configurations.