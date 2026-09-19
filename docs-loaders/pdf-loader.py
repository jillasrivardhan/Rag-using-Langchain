
from langchain_community.document_loaders import TextLoader,PyPDFLoader

loader = PyPDFLoader(file_path="M:\Rag using Langchain\Data\LLM_dictionary.pdf")

docs = loader.load()

# print(docs)
print('-'*50)
print(len(docs))
print('-'*50)
print(docs[0].page_content)
print('-'*50)
print(docs[0])