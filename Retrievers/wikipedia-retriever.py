
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
   top_k_results=3
)

query = "tell me about regression algorithms in ml"

response = retriever.invoke(query)

for i,doc in enumerate(response):
   print(f"{i} ===========")
   print(doc.page_content)
