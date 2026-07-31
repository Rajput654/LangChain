from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text_generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="write a report on the topic {topic}",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="summarize the following reporting 5 points only: {report}",
    input_variables=["report"]
)

parser=StrOutputParser()

chain=prompt1|model|parser|prompt2| model|parser
result=chain.invoke({
    "topic":"dinosaurs"
})
print(result)
chain.get_graph().print_ascii()