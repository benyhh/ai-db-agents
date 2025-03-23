# ai-db-agents
This repository contains implementations of AI agents used to query/retrieve information from various databases. It also contains an application where you can chat with a LLM that has access to fetching movie data from either a SQL database or a vector database.

## Overview
The project demonstrates different approaches to interacting with movie data using language models:

* SQL-based queries using LLM agents
* Vector database retrieval with embeddings
* Retrieval Augmented Generation (RAG) techniques

## Folder Structure

### Explore
The Explore folder contains notebooks and scripts for processing IMDB movie data and exploring different methods for querying and retrieving information:

* Data conversion from CSV to SQLite and vector databases
* Data transformation and preparation
* Batch embedding generation
* Comparison of different query methods (SQL, LangChain, RAG)

### Testing
The Testing folder contains notebooks for testing connections to language models and databases:

* Testing OpenAI API connections
* SQL agent implementation with LangChain
* Vector database embedding and similarity search
* RAG agent implementation

## Data
The movie data can be downloaded from [IMDB datasets](https://datasets.imdbws.com/) and processed using the notebooks in the Explore folder. Sample databases are included in the repository.

## Requirements
* Python 3.8+
* OpenAI API key (for embeddings and language models)
* Libraries: LangChain, ChromaDB, and others specified in the notebooks
* Environment variables set in a .env file

## Usage
1. Set up your environment with the required API keys
2. Explore the data processing workflow in the Explore folder
3. Test component functionality using the Testing notebooks
4. Use the main application to interact with the movie database through natural language

Sample databases are included in the repository, but you can also process the raw IMDB data using the provided notebooks.