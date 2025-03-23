# Explore
This folder contains notebooks and scripts for processing movie data from IMDB and exploring different methods for querying and retrieving information.

## Files
* **01_csv_to_sql_db.ipynb**: Converts raw IMDB CSV data to a SQLite database.

* **02_transform_tabular_data.ipynb**: Performs transformations on the tabular data extracted from the CSV files.

* **03_csv_to_vector_db.ipynb**: Converts CSV data into a vector database format.

* **04_prepare_batch_embedding.ipynb**: Prepares data for batch embedding processing.

* **05_batch_embedding.py**: Python script for generating embeddings in batches through the OpenAI batch embedding API endpoint.

* **06_batch_embedding_to_vector_db.ipynb**: Stores the generated embeddings in a vector database (ChromaDB).

* **07_comparing_methods.ipynb**: Compares different methods for querying data:
  * Simple SQL query generation with LLMs
  * LangChain SQL Agents
  * RAG (Retrieval Augmented Generation) with vector search

## Usage
These notebooks demonstrate a complete workflow from raw data to querying data from different databases. They can be used to understand different approaches to data retrieval with LLMs from both traditional SQL databases and modern vector databases with embeddings.
The notebooks require access to OpenAI API (for embeddings and language models) and proper setup of the local databases (though sample databases are included in the repository).