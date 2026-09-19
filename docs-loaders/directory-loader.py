
from langchain_community.document_loaders import TextLoader,DirectoryLoader,unstructured

loader = DirectoryLoader(path="../Directory",glob="*.txt")

docs = loader.load()

# print(docs)
print('-'*50)
print(docs[0])