from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text_generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

model=ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='give the sentiment of the feedback')

parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negativee \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}

)

classifier_chain=prompt1 | model | parser2

prompt2=PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template='write an appropriate responce to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model|parser1),
    (lambda x:x.sentiment=='negative',prompt3|model|parser1),
    RunnableLambda(lambda x:"could not classify the feedback!")

)
chain=classifier_chain|branch_chain
result=chain.invoke({
    "feedback":"this is the worst restaurant i have ever visited."
})
print(result)
chain.get_graph().print_ascii()