from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel , EmailStr,Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
llm=HuggingFaceEndpoint(
      repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
model= ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str=Field(description="Name of the person.")
    age : int=Field(description="Age of the person.")
    city : str=Field(description="City where the person belong to.")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
template=(
        "You are a helpful assistant. Generate details about a person from {country}.\n\n"
        "IMPORTANT: Return ONLY a valid JSON object with the keys 'name', 'age', and 'city'. "
        "Do NOT return a JSON schema, do NOT include 'properties' or 'required' wrappers, "
        "and do NOT include any markdown formatting or extra text.\n\n"
        "{format_instructions}"
    ),
    input_variables=["country"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain =template|model|parser

res=chain.invoke({"country":"India"})
print(res)