from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

OPEN_AI_KEY=os.getenv("OPEN_AI_KEY")

text= """
Our company leave policy: Employes get 12 sick days per year.
Sick leave resets on January 1st. Days beyond 12 require a medical certificate.
Employees also get 15 casual leaves per year. Casual leave cannot be carried forward.
"""

splitter= RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)

chunks= splitter.create_documents([text])

print(f"Total chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n Chunk {i+1}:")
    print(chunk.page_content)

embeddings= OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPEN_AI_KEY)
vector_store= Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Chunks stored in vector database")

# question= "How many sick days do I get?"
# question = "Can I carry forward my casual leave?"
def ask(question):
    
    # question = "What is the maternity leave policy?"

    retriever= vector_store.as_retriever(search_kwargs={"k":2})
    relevant_chunks= retriever.invoke(question)

    context="\n".join([chunk.page_content for chunk in relevant_chunks])

    prompt= ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.
    Context: {context}
    Question: {question}                                          
    """                                       
    )

    llm= ChatOpenAI(model="gpt-4o",temperature=0,api_key=OPEN_AI_KEY)
    chain=prompt|llm
    response= chain.invoke({"context": context,"question":question})
    return response.content


print(ask("What is the maternity leave policy?"))