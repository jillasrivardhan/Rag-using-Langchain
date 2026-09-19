
from langchain_community.document_loaders import TextLoader
# from langchain_ollama import ChatOllama

loader = TextLoader(file_path="../Data/OFL.txt")

docs = loader.load()

# print(docs)
print(docs[0].page_content)