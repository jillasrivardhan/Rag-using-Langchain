
from langchain_text_splitters import TextSplitter,RecursiveCharacterTextSplitter,Language,CharacterTextSplitter

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
docs = python_splitter.create_documents([PYTHON_CODE])

# print(docs)

for i, doc in enumerate(docs):
    print('-'*20)
    print(f"Chunk {i}: {doc.page_content}")   
    print('-'*20)