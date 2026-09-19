
from langchain_community.document_loaders import TextLoader,DirectoryLoader,unstructured

loader = DirectoryLoader(path="../Directory",glob="*.txt")

docs = loader.load()

# print(docs)
print('-'*50)
print(docs[0])

# in directory if we have many types of files like (pdfs,txt,csv and other) we have to use lazy_load() because it loads the files one by one so that it will be fast,but if we use load() it will try to load all at once so it will take time.