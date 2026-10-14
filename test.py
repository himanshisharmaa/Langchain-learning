from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from argparse import ArgumentParser
import os
from dotenv import load_dotenv
load_dotenv()

ap=ArgumentParser()

ap.add_argument('-q','--question',help="User Question",required=True)
args=vars(ap.parse_args())
question=args['question']

OPEN_AI_KEY=os.getenv("OPEN_AI_KEY")

llm= ChatOpenAI(model="gpt-4o",temperature=0.7,api_key=OPEN_AI_KEY)

prompt=ChatPromptTemplate.from_template("Answer in one sentence: {question}")

chain= prompt|llm

response= chain.invoke({"question":question})

print(response.content)