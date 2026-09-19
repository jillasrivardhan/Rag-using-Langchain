
from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
   model="qwen2.5:3b"
)

prompt = PromptTemplate(
   template="answer the question user asks based on {text} \n {user_input}",
   input_variables=['user_innput','text']
)

parser = StrOutputParser()


loader = WebBaseLoader(web_path="https://medium.com/data-and-beyond/runnable-and-its-types-in-langchain-3b7a1ccfc922")

docs = loader.load()

query = input("enter: ")

chain = prompt | model | parser

response = chain.invoke({'user_input':query,'text':docs[0].page_content})

print(response)


print(docs[0])