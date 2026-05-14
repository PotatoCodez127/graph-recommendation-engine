# Streaming Service Recommendation Engine (GraphRAG)

## Overview
This project demonstrates how to use Knowledge Graphs and GraphRAG to solve complex, multi-hop reasoning problems that standard Vector RAG fails to understand. 

Instead of searching for text similarity, this engine maps explicit relationships between entities (Users, Movies, Directors, Genres) and uses an LLM to traverse these connections to provide highly logical recommendations.

## The Problem Solved
Standard RAG relies on vector similarity, making it terrible at "Missing Link" problems (e.g., "Recommend a movie directed by the person who directed Inception, but in the Comedy genre"). Graph databases solve this by explicitly linking data nodes, allowing the AI to traverse paths to find exact, factual connections.

## Tech Stack
* **Python 3.10+**
* **NetworkX:** A Python library for the creation, manipulation, and study of complex networks (our local Graph DB substitute).
* **Ollama Cloud:** For the LLM reasoning phase.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment.
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your API key:
   `OLLAMA_API_KEY=your_api_key_here`

## Usage
Run the main graph engine:
`python graph_engine.py`