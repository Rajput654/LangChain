from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
import os
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

prompt2=PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)
chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)
print(chain.invoke({'topic':'AI'}))