

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

documents = [

    Document(
        page_content="""
        The Toyota Camry is a midsize sedan known for its reliability,
        comfortable interior, and smooth driving experience. It is suitable
        for both daily commuting and long-distance travel. The Camry offers
        a spacious cabin and modern safety features.
        """,
        metadata={
            "brand": "Toyota",
            "model": "Camry",
            "year": 2025,
            "category": "Sedan",
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "price": "$28,400"
        }
    ),

    Document(
        page_content="""
        The BMW 3 Series is a luxury sports sedan that combines performance,
        premium interiors, and advanced technology. It provides responsive
        handling and a powerful engine, making it popular among drivers who
        want both comfort and sporty performance.
        """,
        metadata={
            "brand": "BMW",
            "model": "3 Series",
            "year": 2025,
            "category": "Luxury Sedan",
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "price": "$45,000"
        }
    ),

    Document(
        page_content="""
        The Ford Mustang is a famous American sports car recognized for its
        powerful engine and aggressive styling. It offers strong acceleration,
        sporty handling, and a driver-focused experience. The Mustang is
        designed for people who enjoy performance-oriented cars.
        """,
        metadata={
            "brand": "Ford",
            "model": "Mustang",
            "year": 2025,
            "category": "Sports Car",
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "price": "$31,920"
        }
    ),

    Document(
        page_content="""
        The Tesla Model 3 is an electric sedan designed for efficient
        transportation and modern technology. It provides instant electric
        acceleration, a minimalist interior, and a long driving range.
        The vehicle also includes advanced driver-assistance features.
        """,
        metadata={
            "brand": "Tesla",
            "model": "Model 3",
            "year": 2025,
            "category": "Electric Sedan",
            "fuel_type": "Electric",
            "transmission": "Automatic",
            "price": "$42,490"
        }
    ),

    Document(
        page_content="""
        The Jeep Wrangler is an off-road focused SUV known for its rugged
        design and strong four-wheel-drive capability. It is designed for
        outdoor adventures and challenging terrain. The Wrangler also offers
        removable roof and doors on certain configurations.
        """,
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
]

vector_store = Chroma.from_documents(
   documents=documents,
   embedding=OllamaEmbeddings(model="qwen3-embedding:0.6b")
)

response = vector_store.as_retriever( search_type="mmr",search_kwargs={"k": 3, "lambda_mult": 1})

query = "what are the different types of suv cars are there?"

result = response.invoke(query)

for i,doc in enumerate(result):
   print(f"{i} ===========")
   print(doc.page_content)
