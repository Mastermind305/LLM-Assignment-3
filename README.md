# LLM-Assignment-3
Below is a combined README file that you can use for your GitHub repository to document the three assignments (3.1 RAG, 3.2 Multi-Agent System, and 3.3 Parameter-Efficient Fine-Tuning). This README provides an overview, setup instructions, usage details, and additional information for each assignment.

---

# Assignment 3: AI Systems and Techniques

This repository contains the implementation of three assignments from Assignment 3, focusing on advanced AI techniques:

- **3.1: Retrieval-Augmented Generation (RAG)** - A system to crawl Wikipedia, retrieve relevant context using FAISS, and generate answers with a transformer model.
- **3.2: Multi-Agent System** - An asynchronous multi-agent framework using LLMs for task delegation and collaboration.
- **3.3: Parameter-Efficient Fine-Tuning** - Fine-tuning transformer models for sentiment analysis with various efficiency techniques.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Assignments](#assignments)
  - [3.1: Retrieval-Augmented Generation (RAG)](#31-retrieval-augmented-generation-rag)
  - [3.2: Multi-Agent System](#32-multi-agent-system)
  - [3.3: Parameter-Efficient Fine-Tuning](#33-parameter-efficient-fine-tuning)
- [Usage](#usage)


## Overview

This repository showcases three distinct AI projects developed as part of Assignment 3:

- **RAG (3.1)** implements a question-answering system that leverages web crawling and vector search to provide context-aware responses.
- **Multi-Agent System (3.2)** demonstrates a collaborative agent framework for task planning, validation, summarization, and answering using LLMs.
- **Parameter-Efficient Fine-Tuning (3.3)** explores different fine-tuning strategies (full tuning, LoRA, frozen backbone, and gradual unfreezing) for sentiment analysis on the IMDb dataset.

## Prerequisites

- Python 3.11 or higher
- Git (for cloning the repository)
- Required libraries (install via `requirements.txt` or individual commands below)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/assignment-3-ai.git
   cd assignment-3-ai
   ```

2. Install dependencies:
   - For all assignments, create a virtual environment and install requirements:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     pip install -r requirements.txt
     ```
   - Alternatively, install specific dependencies:
     - **3.1 (RAG):**
       ```bash
       pip install datasets sentence-transformers faiss-cpu transformers gradio beautifulsoup4 requests rouge-score nltk
       ```
     - **3.2 (Multi-Agent System):**
       ```bash
       pip install openai
       ```
     - **3.3 (Parameter-Efficient Fine-Tuning):**
       ```bash
       pip install torch transformers datasets numpy pandas matplotlib seaborn psutil scikit-learn
       ```

3. Set up environment variables (for 3.2):
   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Ensure the `.env` file is added to `.gitignore` to keep your API key secure.

4. Verify installation:
   - Run `python --version` and check for the required libraries in your Python environment.

## Assignments

### 3.1: Retrieval-Augmented Generation (RAG)

**Description:** This script (`rag.py`) builds a RAG system that crawls a Wikipedia page (e.g., SpaceX), chunks the text, indexes it with FAISS, and uses a BART model to generate answers to questions based on retrieved context. It includes a Gradio interface for interactive use and evaluates performance with ROUGE and BLEU scores.

**Files:**
- `rag.py`

**Features:**
- Web crawling using BeautifulSoup
- Text chunking and FAISS indexing
- Question-answering with BART
- Gradio UI and evaluation metrics

### 3.2: Multi-Agent System

**Description:** This script (`multi_agent_system.py`) implements an asynchronous multi-agent system with four agents (Planner, Validator, Summarizer, Answer) that collaborate to process a user task using OpenAI's API. Agents communicate via queues and handle planning, validation, summarization, and final response generation.

**Files:**
- `multi_agent_system.py`

**Features:**
- Asynchronous agent communication
- Task delegation and validation
- LLM-based processing
- Dynamic user input support

### 3.3: Parameter-Efficient Fine-Tuning

**Description:** This Jupyter notebook (`fine-tune-transformer-for-sentimental-analysis.ipynb`) explores fine-tuning a DistilBERT model for sentiment analysis on the IMDb dataset using four approaches: full fine-tuning, LoRA, frozen backbone, and gradual unfreezing. It includes training metrics, memory usage, and evaluation plots.

**Files:**
- `fine-tune-transformer-for-sentimental-analysis.ipynb`

**Features:**
- Custom dataset class for IMDb
- Multiple fine-tuning strategies
- Comprehensive metrics (accuracy, loss, memory)
- Visualization of results

## Usage

### 3.1: Retrieval-Augmented Generation (RAG)
1. Run the script:
   ```bash
   python rag.py
   ```
2. Open the Gradio interface in your browser (link provided in the terminal).
3. Enter a Wikipedia URL and question, then submit to get an answer.
4. View evaluation results for predefined SpaceX questions in the console.

### 3.2: Multi-Agent System
1. Ensure the `OPENAI_API_KEY` is set in your `.env` file.
2. Run the script:
   ```bash
   python multi_agent_system.py
   ```
3. Enter a task/question when prompted (or use the default).
4. Observe the agent logs and final output in the console.

### 3.3: Parameter-Efficient Fine-Tuning
1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook fine-tune-transformer-for-sentimental-analysis.ipynb
   ```
2. Run all cells to load the dataset, train models, and visualize results.
3. Analyze the plots and metrics for each fine-tuning approach.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Description of changes"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request with a clear description of your changes.


---


-
