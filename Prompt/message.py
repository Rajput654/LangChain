from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

model=ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="you are a good assistant"),
    HumanMessage(content="what is the capital of india?")
]
response=model.invoke(messages)
messages.append(AIMessage(content=response.content))
print(response.content)
print(messages)
