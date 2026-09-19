
from langchain_community.document_loaders import TextLoader,CSVLoader

loader = CSVLoader(file_path="M:\Rag using Langchain\Data\Iris.csv")

docs = loader.load()

# print(docs)
print('-'*50)
print(docs[0].page_content)
print('-'*50)
print(len(docs))