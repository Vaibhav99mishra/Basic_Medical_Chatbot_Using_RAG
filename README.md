# Basic_Medical_Chatbot_Using_RAG

This project implements a Retrieval-Augmented Generation (RAG) based medical chatbot using a single Jupyter Notebook. The chatbot aims to provide accurate and contextually relevant responses to medical queries by combining retrieval from a vector database with generative capabilities.

## Project Overview

Medical chatbots can play a crucial role in providing healthcare information, symptom assessment, and wellness advice, particularly for remote areas or during non-office hours. This project leverages a RAG-based approach to build a medical chatbot that retrieves relevant contexts and generates responses, enhancing user interaction and information accuracy.

### Key Components of the Project

1. **Data Gathering and Preparation**:
   - Load and prepare a medical dataset from Hugging Face (e.g., PubMed QA) to train the chatbot.
   - Preprocess and chunk context data to fit within the model's embedding limitations.

2. **Vector Database Creation**:
   - Create dense and sparse embeddings of medical contexts using Sentence Transformers and SpladeEncoder models.
   - Store embeddings in a Pinecone vector database to enable fast and accurate retrieval.

3. **Retrieval-Augmented Generation (RAG) Pipeline**:
   - Build a RAG pipeline using LangChain to combine retrieval from Pinecone with text generation using OpenAI’s GPT model.
   - Queries are passed to the RAG pipeline, which retrieves relevant contexts and generates a response.

4. **Evaluation**:
   - Use the RAGAS evaluation metrics to assess the quality of responses based on **context recall**, **context precision**, **faithfulness**, **answer relevancy**, **answer similarity**, and **answer correctness**.
   - Visualize evaluation results for better insights into model performance.

## Project Flow

1. **Import Libraries**: Import all necessary Python libraries such as Hugging Face datasets, Pinecone, Sentence Transformers, LangChain, and RAGAS.

2. **Data Loading and Preprocessing**:
   - Load a dataset from Hugging Face.
   - Chunk contexts into manageable sizes.

3. **Vector Database Setup**:
   - Generate dense and sparse embeddings for each context.
   - Upload embeddings to Pinecone for efficient retrieval.

4. **RAG Pipeline Setup**:
   - Set up LangChain with Pinecone for context retrieval.
   - Use OpenAI’s GPT model to generate answers based on retrieved contexts.

5. **Evaluation**:
   - Test the chatbot's responses on sample queries.
   - Calculate performance metrics using RAGAS.

## Example Usage

### Question and Answering

```python
# Ask a question using the RAG pipeline
query = "What are the symptoms of diabetes?"
result = qa.invoke(query)
print("Generated Response:", result['result'])
```

### Evaluation Example

```python
# Evaluate RAG pipeline with sample queries and responses
eval_data = [...]  # Prepare sample evaluation data
result = evaluate(
    dataset=ragas_eval,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print("Evaluation Results:", result)
```

## Setup Instructions

1. **Install Dependencies**:
   Make sure to install all necessary Python libraries. Use the following command to install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys**:
   - Set up an API key for Pinecone in your environment variables.
   - Set up an API key for OpenAI in your environment variables if using GPT for generation.

3. **Run the Notebook**:
   Open the Jupyter Notebook and follow the code sections to execute each part of the workflow.

## Dependencies

- Python 3.8+
- `transformers`
- `datasets`
- `pinecone-client`
- `sentence-transformers`
- `langchain`
- `ragas`
- `matplotlib`
- `seaborn`

## Limitations

- The model's responses are limited to the quality and scope of the dataset provided.
- Requires access to Pinecone and OpenAI API for the retrieval and generation process.

