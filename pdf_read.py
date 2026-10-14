from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_KEY=os.getenv("OPEN_AI_KEY")

ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",help="Add the input file in pdf format",required=True)
args=vars(ap.parse_args())
pdf_file=args["input"]

loader=PyPDFLoader(pdf_file)
documents=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
chunks=splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

embeddings=OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPEN_AI_KEY)
vector_store=Chroma.from_documents(documents=chunks,embedding=embeddings)
print("Chunks stored in vector database")

prompt= ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context below.
    Context: {context}
    Question: {question}
    """
)

llm= ChatOpenAI(model="gpt-4o", temperature=0,api_key=OPEN_AI_KEY)
chain= prompt|llm

def ask(question):
    retriever= vector_store.as_retriever(search_kwargs={"k":2})
    relevant_chunks= retriever.invoke(question)

    context= "\n".join([chunk.page_content for chunk in relevant_chunks])
    response= chain.invoke({"context": context,"question":question})
    return response.content

print(ask("How many casual leaves are there"))