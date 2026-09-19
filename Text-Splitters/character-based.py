
from langchain_text_splitters import TextSplitter,RecursiveCharacterTextSplitter,Language,CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("OFL.txt")

docs = loader.load()

splitter = CharacterTextSplitter(
   chunk_size = 200,
   chunk_overlap = 10
)

res = splitter.split_documents(docs)

for i, doc in enumerate(res):
    print('-'*20)
    print(f"Chunk {i}: {doc.page_content}")   
    print('-'*20)

# print(res[0].page_content)
# print(res)   
# print(len(res)) 