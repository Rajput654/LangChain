
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='Answer the following question \n {question} from the following text -\n {text}',
    input_variables=['question','text']
)
parser=StrOutputParser()

url='https://fiverr-res.cloudinary.com/image/upload/v1/attachments/generic_asset/asset/1e3a656ddbd3c8d61d6933c7372975be-1784794566444/Freelancer%20Annotation%20Onboarding%20Guide_FINAL.pdf'
loader=WebBaseLoader(url)

docs=loader.load()
chain=prompt | model |parser
print(chain.invoke({'question':'what is the topic that is discussed here?','text':docs[0].metadata}))