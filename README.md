# 🧠 RAG Using LangChain

A hands-on learning repository for understanding and implementing **Retrieval-Augmented Generation (RAG)** concepts using **LangChain, Ollama, embeddings, vector databases, document loaders, text splitters, and retrievers**.

This project focuses on learning the individual building blocks behind modern RAG systems before combining them into complete AI applications.

---

## 📌 Overview

Large Language Models can generate powerful responses, but they do not automatically have access to your private documents or custom datasets.

**Retrieval-Augmented Generation (RAG)** solves this problem by retrieving relevant information from an external knowledge source and providing that information to an LLM before generating a response.

The general RAG workflow is:

```text
                ┌──────────────────┐
                │   Documents      │
                │ PDF / TXT / CSV  │
                │ Web / Directory  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Document Loader │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Text Splitter   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Embeddings    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Vector Store    │
                │ Chroma / FAISS   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │    Retriever     │
                │ Similarity / MMR │
                │   Multi-Query    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │       LLM        │
                │     Ollama       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     Response     │
                └──────────────────┘
```

---

# 🎯 Learning Objectives

This repository is designed to help you understand:

* What RAG is
* Why RAG is useful
* LangChain document loaders
* Loading TXT files
* Loading CSV files
* Loading PDF files
* Loading directories
* Loading web content
* Document objects
* Metadata
* Text splitting
* Character-based splitting
* Structured text splitting
* Embeddings
* Vector databases
* Chroma
* FAISS
* Similarity search
* Similarity search with scores
* Maximum Marginal Relevance (MMR)
* Multi-Query Retrieval
* Wikipedia retrieval
* Ollama embeddings
* Ollama chat models
* Building blocks of RAG systems

---

# 🛠️ Technologies Used

| Technology          | Purpose                         |
| ------------------- | ------------------------------- |
| Python              | Programming language            |
| LangChain           | LLM application framework       |
| LangChain Core      | Core abstractions               |
| LangChain Community | Loaders and integrations        |
| LangChain Ollama    | Ollama model integration        |
| LangChain Chroma    | Chroma vector store integration |
| Ollama              | Local LLM and embedding models  |
| Chroma              | Vector database                 |
| FAISS               | Vector similarity search        |
| PyPDF               | PDF document processing         |
| Wikipedia           | Knowledge retrieval             |
| Pydantic            | Data validation where required  |

---

# 📂 Project Structure

```text
Rag-using-Langchain/
│
├── Data/
│   ├── Iris.csv
│   └── LLM_dictionary.pdf
│
├── Directory/
│   └── OFL.txt
│
├── docs-loaders/
│   ├── csv-loader.py
│   ├── directory-loader.py
│   ├── pdf-loader.py
│   ├── text-loader.py
│   └── webbased-loader.py
│
├── Text-Splitters/
│   ├── OFL.txt
│   ├── character-based.py
│   ├── structured-character.py
│   └── structured.py
│
├── Retrievers/
│   ├── MMR.py
│   ├── Multi-query.py
│   ├── vector-store.py
│   └── wikipedia-retriever.py
│
├── vector-stores/
│   ├── FAISS.py
│   ├── chroma.py
│   └── chroma_langchain_db/
│
└── requirements.txt
```

---

# 📚 1. Document Loaders

Document loaders are responsible for bringing external data into a format that LangChain can process.

A typical loader workflow is:

```text
External Data
     │
     ▼
Document Loader
     │
     ▼
LangChain Document
     │
     ├── page_content
     │
     └── metadata
```

Each loaded document generally contains:

```python
Document(
    page_content="...",
    metadata={}
)
```

---

## 📄 Text Loader

File:

```text
docs-loaders/text-loader.py
```

The text loader demonstrates how to load a `.txt` file into LangChain.

Example:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("Directory/OFL.txt")

docs = loader.load()

print(docs)
```

Useful when working with:

* `.txt` files
* Notes
* Documentation
* Plain-text datasets
* Knowledge bases

---

# 📊 CSV Loader

File:

```text
docs-loaders/csv-loader.py
```

The CSV loader demonstrates how structured tabular data can be loaded into LangChain documents.

Useful for:

* CSV datasets
* Business data
* Product information
* Customer records
* Structured datasets

---

# 📕 PDF Loader

File:

```text
docs-loaders/pdf-loader.py
```

The PDF loader demonstrates loading information from PDF documents.

The repository includes:

```text
Data/LLM_dictionary.pdf
```

This can be used as an example knowledge source.

Typical PDF RAG workflow:

```text
PDF
 │
 ▼
PDF Loader
 │
 ▼
Documents
 │
 ▼
Text Splitter
 │
 ▼
Embeddings
 │
 ▼
Vector Store
```

---

# 📁 Directory Loader

File:

```text
docs-loaders/directory-loader.py
```

Directory loaders are useful when your knowledge base contains multiple files.

Instead of loading every file individually:

```text
Knowledge Base
│
├── document1.txt
├── document2.txt
├── document3.txt
└── document4.txt
```

a directory loader can process the collection of files.

---

# 🌐 Web-Based Loader

File:

```text
docs-loaders/webbased-loader.py
```

This example demonstrates retrieving content from a web page using a web loader and passing the retrieved content to an Ollama model.

Conceptually:

```text
Web Page
   │
   ▼
Web Loader
   │
   ▼
Page Content
   │
   ▼
Prompt
   │
   ▼
Ollama LLM
   │
   ▼
Answer
```

---

# ✂️ 2. Text Splitters

Large documents should usually be divided into smaller chunks before generating embeddings.

The repository contains multiple text-splitting examples.

```text
Text-Splitters/
├── character-based.py
├── structured-character.py
└── structured.py
```

---

## Why Do We Need Text Splitting?

Imagine a document contains:

```text
100,000 words
```

Sending the entire document for every query is inefficient.

Instead:

```text
Large Document
      │
      ▼
   Splitter
      │
      ├── Chunk 1
      ├── Chunk 2
      ├── Chunk 3
      ├── Chunk 4
      └── ...
```

Each chunk can then be embedded and stored in a vector database.

---

## Character-Based Splitting

Character-based splitting divides text according to character boundaries.

Important concepts include:

* Chunk size
* Chunk overlap
* Separators

Example concept:

```text
Original Document
       │
       ▼
┌────────────────┐
│ Chunk 1        │
└────────────────┘
       │ overlap
       ▼
┌────────────────┐
│ Chunk 2        │
└────────────────┘
```

---

## Chunk Size

`chunk_size` controls approximately how large each chunk should be.

For example:

```python
chunk_size=500
```

means the splitter attempts to create chunks around that size.

---

## Chunk Overlap

`chunk_overlap` keeps some information from the previous chunk.

Example:

```python
chunk_size=500
chunk_overlap=50
```

Conceptually:

```text
Chunk 1:
[--------------------500--------------------]

                     overlap
                         ↓

Chunk 2:
                    [--------------------500--------------------]
```

Overlap helps preserve context across chunk boundaries.

---

# 🧩 3. Embeddings

Embeddings convert text into numerical vectors.

For example:

```text
"Machine learning is a branch of AI"
                 │
                 ▼
          Embedding Model
                 │
                 ▼
[0.12, -0.43, 0.81, 0.27, ...]
```

Semantically similar text should generally produce vectors that are closer together in embedding space.

This makes embeddings useful for semantic search.

---

# 🤖 Ollama Embeddings

This repository uses Ollama embeddings in several examples.

Example:

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)
```

The embedding model is then used to convert documents and queries into vectors.

---

# 🗄️ 4. Vector Stores

Vector stores are databases or indexes designed to store and search vector representations.

This project demonstrates:

* Chroma
* FAISS

The general workflow is:

```text
Documents
    │
    ▼
Embeddings
    │
    ▼
Vectors
    │
    ▼
Vector Store
```

When a user asks a question:

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
Vector Search
    │
    ▼
Relevant Documents
```

---

# 🟢 Chroma

File:

```text
vector-stores/chroma.py
```

The project demonstrates:

* Creating a Chroma vector store
* Adding documents
* Generating embeddings
* Similarity search
* Similarity search with scores
* Updating documents

Example:

```python
from langchain_chroma import Chroma

vectors = Chroma(
    collection_name="cars_data",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
```

The repository also contains a local Chroma database:

```text
vector-stores/chroma_langchain_db/
```

---

# 🔵 FAISS

File:

```text
vector-stores/FAISS.py
```

FAISS can be used for efficient similarity search over vector representations.

The example demonstrates storing document embeddings and performing similarity-based retrieval.

---

# 🔎 5. Similarity Search

Similarity search attempts to find documents that are semantically close to a user's query.

Example query:

```text
Which cars are suitable for off-road driving?
```

The vector database compares the query embedding with stored document embeddings.

```text
Query
  │
  ▼
Embedding
  │
  ▼
Vector Search
  │
  ├── Jeep Wrangler
  ├── Ford Mustang
  └── Toyota Camry
```

The most relevant documents can then be returned.

---

# 📊 Similarity Search With Scores

The repository also demonstrates:

```python
similarity_search_with_score()
```

This returns documents along with a similarity/distance score provided by the vector store.

This can be useful when you want to inspect how closely retrieved documents match a query.

---

# 🎯 6. Retrievers

Retrievers are responsible for finding relevant documents from a knowledge source.

This repository demonstrates multiple retrieval strategies.

```text
Retrievers/
├── MMR.py
├── Multi-query.py
├── vector-store.py
└── wikipedia-retriever.py
```

---

# 🔀 Maximum Marginal Relevance — MMR

File:

```text
Retrievers/MMR.py
```

MMR stands for:

> Maximum Marginal Relevance

MMR attempts to balance:

```text
Relevance
    +
Diversity
```

Instead of simply returning several highly similar documents, MMR can help retrieve documents that are both relevant and sufficiently different from each other.

Example:

```text
User Query
    │
    ▼
Vector Search
    │
    ▼
MMR Retrieval
    │
    ├── Relevant Document 1
    ├── Relevant Document 2
    └── Relevant Document 3
```

The example in this repository uses:

```python
response = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 1
    }
)
```

---

# 🔄 Multi-Query Retrieval

File:

```text
Retrievers/Multi-query.py
```

Users can ask the same question in many different ways.

For example:

```text
Which cars are good for outdoor adventures?

Which vehicles are suitable for off-road activities?

What cars are designed for outdoor driving?
```

A Multi-Query Retriever can use an LLM to generate alternative versions of a query and retrieve documents for those variations.

Conceptually:

```text
                User Query
                    │
                    ▼
              LLM generates
            alternative queries
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    Query 1       Query 2      Query 3
       │            │            │
       └────────────┼────────────┘
                    ▼
              Vector Store
                    │
                    ▼
           Retrieved Documents
```

The repository uses:

```python
MultiQueryRetriever.from_llm()
```

with:

* Chroma
* Ollama embeddings
* `qwen2.5:3b`

---

# 🌎 Wikipedia Retriever

File:

```text
Retrievers/wikipedia-retriever.py
```

This example demonstrates retrieving information from Wikipedia through a LangChain retriever.

It introduces the idea that a RAG system does not necessarily need to use only a local vector database.

Possible knowledge sources include:

```text
Local Files
     │
     ├── TXT
     ├── PDF
     └── CSV

External Sources
     │
     ├── Web
     └── Wikipedia
```

---

# 🏗️ RAG Architecture

The concepts in this repository can eventually be combined into a complete RAG pipeline.

```text
                 ┌───────────────┐
                 │ User Question │
                 └───────┬───────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Query Embedding │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Vector Database │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Retriever    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │Relevant Context │
                └────────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Prompt + Context     │
              │ + User Question      │
              └──────────┬───────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Ollama LLM    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Final Response  │
                └─────────────────┘
```

---

# 🖥️ Local LLM Setup

This project uses **Ollama** so that the LLM and embedding models can run locally.

Install Ollama on your system and make sure the Ollama service is running.

Pull the models used by the examples:

```bash
ollama pull qwen2.5:3b
```

For embeddings:

```bash
ollama pull qwen3-embedding:0.6b
```

Verify installed models:

```bash
ollama list
```

> Model availability and hardware requirements can vary. If a model is too large for your machine, use another compatible Ollama model and update the model name in the Python files.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/jillasrivardhan/Rag-using-Langchain.git
```

Move into the project:

```bash
cd Rag-using-Langchain
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv rag
```

Activate it:

```bash
rag\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv rag
```

Activate:

```bash
source rag/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

The project includes dependencies for:

```text
langchain
langchain-core
langchain-community
langchain-ollama
langchain-chroma
wikipedia
pypdf
```

Install everything using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Examples

Each directory contains independent learning examples.

### Document Loaders

```bash
python docs-loaders/text-loader.py
```

```bash
python docs-loaders/csv-loader.py
```

```bash
python docs-loaders/pdf-loader.py
```

```bash
python docs-loaders/directory-loader.py
```

```bash
python docs-loaders/webbased-loader.py
```

### Text Splitters

```bash
python Text-Splitters/character-based.py
```

```bash
python Text-Splitters/structured-character.py
```

```bash
python Text-Splitters/structured.py
```

### Retrievers

```bash
python Retrievers/MMR.py
```

```bash
python Retrievers/Multi-query.py
```

```bash
python Retrievers/vector-store.py
```

```bash
python Retrievers/wikipedia-retriever.py
```

### Vector Stores

```bash
python vector-stores/chroma.py
```

```bash
python vector-stores/FAISS.py
```

---

# 🧪 Example Dataset

The repository contains sample car documents with metadata such as:

```text
Brand
Model
Year
Category
Fuel Type
Transmission
Price
```

For example:

```python
Document(
    page_content="The Jeep Wrangler is an off-road focused SUV...",
    metadata={
        "brand": "Jeep",
        "model": "Wrangler",
        "year": 2025,
        "category": "SUV",
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "price": "$33,690"
    }
)
```

This makes the dataset useful for learning:

* Semantic search
* Metadata
* Vector databases
* Retrieval
* Similarity search
* MMR
* Multi-query retrieval

---

# 🧠 Key LangChain Concepts Learned

By completing this repository, you will work with:

### Documents

```python
Document(
    page_content="...",
    metadata={}
)
```

### Document Loaders

Load external information into LangChain.

### Text Splitters

Break large documents into manageable chunks.

### Embeddings

Convert text into numerical vector representations.

### Vector Stores

Store and search embeddings.

### Retrievers

Retrieve relevant documents based on a query.

### LLMs

Generate responses using retrieved context.

---

# 🔥 RAG vs Traditional LLM

| Traditional LLM                             | RAG                                                   |
| ------------------------------------------- | ----------------------------------------------------- |
| Uses model knowledge                        | Uses model + external knowledge                       |
| Limited access to private data              | Can retrieve private/custom data                      |
| May hallucinate when information is missing | Retrieved context can ground responses                |
| Knowledge depends on model                  | Knowledge can be updated through the retrieval source |
| No retrieval layer                          | Contains a retrieval pipeline                         |

---

# 🚀 Possible Future Improvements

This repository can be extended into complete RAG applications.

### Beginner

* Build a PDF question-answering system
* Add more document types
* Create a simple CLI chatbot
* Experiment with different chunk sizes
* Compare different embedding models

### Intermediate

* Build a Streamlit RAG application
* Add conversation history
* Add metadata filtering
* Add source-document citations
* Compare Chroma and FAISS
* Experiment with different retrieval strategies

### Advanced

* Build a production-style RAG pipeline
* Add hybrid search
* Add reranking
* Add query rewriting
* Add document ingestion pipelines
* Add evaluation metrics
* Add LangSmith tracing
* Add persistent conversation memory
* Deploy the application

---

# 🧪 Recommended Learning Path

If you are learning RAG from scratch, follow this order:

```text
1. Document Loaders
        ↓
2. Documents & Metadata
        ↓
3. Text Splitters
        ↓
4. Embeddings
        ↓
5. Vector Stores
        ↓
6. Similarity Search
        ↓
7. Retrievers
        ↓
8. MMR
        ↓
9. Multi-Query Retrieval
        ↓
10. RAG Pipeline
        ↓
11. RAG Application
```

This progression makes it easier to understand how each individual component contributes to a complete RAG system.

---

# ⚠️ Troubleshooting

## Ollama Connection Error

Make sure Ollama is running.

Check:

```bash
ollama list
```

If the command does not work, start/install Ollama before running the examples.

---

## Model Not Found

If you see an error similar to:

```text
model not found
```

pull the required model:

```bash
ollama pull qwen2.5:3b
```

or:

```bash
ollama pull qwen3-embedding:0.6b
```

---

## FileNotFoundError

If a loader reports:

```text
FileNotFoundError
```

check that the file path is correct relative to the directory from which you execute the Python script.

For example:

```text
Rag-using-Langchain/
│
├── Directory/
│   └── OFL.txt
│
└── docs-loaders/
    └── text-loader.py
```

You may need to use:

```python
TextLoader("Directory/OFL.txt")
```

rather than an incorrect relative path.

---

## LangChain Import Errors

LangChain is actively evolving, and integrations can move between packages.

If an import fails:

1. Check the package installed in your environment.
2. Upgrade the required LangChain integration.
3. Check the current package documentation.
4. Prefer the dedicated integration package when one exists.

For example, the project uses:

```python
from langchain_ollama import ChatOllama
```

and:

```python
from langchain_ollama import OllamaEmbeddings
```

---

# 🎓 What This Project Teaches

This repository is more than a single RAG application.

It breaks the RAG ecosystem into individual components so you can understand what happens at every stage:

```text
                    RAG
                     │
       ┌─────────────┼─────────────┐
       │             │             │
   Ingestion      Retrieval     Generation
       │             │             │
       ▼             ▼             ▼
   Loaders       Embeddings       LLM
       │             │
       ▼             ▼
   Splitters    Vector Store
                     │
                     ▼
                 Retriever
```

Understanding these components individually makes it much easier to build larger RAG applications later.

---

# 🌟 Future Project Ideas

After completing this repository, you can build:

* 📄 PDF Chatbot
* 📚 Personal Knowledge Assistant
* 🧑‍💻 Code Documentation RAG
* 🏫 Educational Document Assistant
* 📊 CSV Data Q&A System
* 🌐 Website Question-Answering System
* 🏢 Company Knowledge Base
* 📖 Research Paper Assistant
* 🧠 Personal AI Memory System
* 🤖 Multi-document RAG Chatbot

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve this learning repository:

1. Fork the repository.
2. Create a new branch.
3. Add your improvement.
4. Test your changes.
5. Commit your changes.
6. Push the branch.
7. Open a Pull Request.

---

# 📄 License

This repository is intended primarily for educational and learning purposes.

Individual datasets, documents, fonts, or third-party resources included in the repository may have their own licenses. Check the relevant source/license files before redistributing those materials.

---

# 👨‍💻 Author

**Jilla Srivardhan**

Learning and building projects around:

* Generative AI
* LangChain
* RAG
* Ollama
* LLM Applications
* Python
* Machine Learning

---

# ⭐ Support

If this repository helped you understand **RAG and LangChain**, consider giving the repository a ⭐ on GitHub.

Keep learning, keep building, and keep experimenting with AI. 🚀
