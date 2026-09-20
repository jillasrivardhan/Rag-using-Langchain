from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# --------------------------------------------------
# 1. Create documents
# --------------------------------------------------

documents = [

    Document(
        page_content="""
        The Toyota Camry is a midsize sedan known for its reliability,
        comfortable interior, and smooth driving experience.
        """,
        metadata={
            "brand": "Toyota",
            "model": "Camry",
            "category": "Sedan",
            "fuel_type": "Petrol"
        }
    ),

    Document(
        page_content="""
        The BMW 3 Series is a luxury sports sedan that combines
        performance, premium interiors, and advanced technology.
        """,
        metadata={
            "brand": "BMW",
            "model": "3 Series",
            "category": "Luxury Sedan",
            "fuel_type": "Petrol"
        }
    ),

    Document(
        page_content="""
        The Ford Mustang is a famous American sports car recognized
        for its powerful engine, aggressive styling, strong acceleration,
        and sporty handling.
        """,
        metadata={
            "brand": "Ford",
            "model": "Mustang",
            "category": "Sports Car",
            "fuel_type": "Petrol"
        }
    ),

    Document(
        page_content="""
        The Tesla Model 3 is an electric sedan designed for efficient
        transportation and modern technology. It provides instant
        electric acceleration and a long driving range.
        """,
        metadata={
            "brand": "Tesla",
            "model": "Model 3",
            "category": "Electric Sedan",
            "fuel_type": "Electric"
        }
    ),

    Document(
        page_content="""
        The Jeep Wrangler is an off-road focused SUV known for its
        rugged design and strong four-wheel-drive capability. It is
        designed for outdoor adventures and challenging terrain.
        """,
        metadata={
            "brand": "Jeep",
            "model": "Wrangler",
            "category": "SUV",
            "fuel_type": "Petrol"
        }
    )
]


# --------------------------------------------------
# 2. Create embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# --------------------------------------------------
# 3. Create vector database
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="car_collection"
)


# --------------------------------------------------
# 4. Create LLM
# --------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)


# --------------------------------------------------
# 5. Create MultiQueryRetriever
# --------------------------------------------------

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    llm=llm
)


# --------------------------------------------------
# 6. Ask a question
# --------------------------------------------------

query = "Which cars are good for outdoor adventures?"

results = retriever.invoke(query)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

for i, doc in enumerate(results, start=1):

    print(f"\n--- Document {i} ---")

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)