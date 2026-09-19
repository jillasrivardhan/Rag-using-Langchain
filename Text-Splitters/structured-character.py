
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

# --------------------------------------------------
# Markdown splitter
# --------------------------------------------------

markdown_text = """
# 🦜️🔗 LangChain

⚡ Building applications with LLMs through composability ⚡

## What is LangChain?

# Hopefully this code block isn't split
LangChain is a framework for...

As an open-source project in a rapidly developing field, we are extremely open to contributions.
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
md_docs = md_splitter.create_documents([markdown_text])

for i, doc in enumerate(md_docs):
    print('-'*20)
    print(f"Chunk {i}: {doc.page_content}")   
    print('-'*20)


# --------------------------------------------------
# HTML splitter
# --------------------------------------------------

html_text = """
<!DOCTYPE html>
<html>
    <head>
        <title>🦜️🔗 LangChain</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            h1 {
                color: darkblue;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🦜️🔗 LangChain</h1>
            <p>⚡ Building applications with LLMs through composability ⚡</p>
        </div>
        <div>
            As an open-source project in a rapidly developing field, we are extremely open to contributions.
        </div>
    </body>
</html>
"""

html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML, chunk_size=60, chunk_overlap=0
)
html_docs = html_splitter.create_documents([html_text])

for i, doc in enumerate(html_docs):
    print('-'*20)
    print(f"Chunk {i}: {doc.page_content}")   
    print('-'*20)
