from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
import os

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="write a summary for the following poem - \n {poem}",
    input_variables=["poem"]
)
parser=StrOutputParser()

loader=TextLoader("C:\\Users\\SANSKAR\\Desktop\\LangChain\\Document-loader\\cricket.txt",encoding='utf-8')
docs=loader.load()

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain=prompt|model|parser

print(chain.invoke({'poem':docs[0].page_content}))