from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.document_loaders import WebBaseLoader,PyPDFLoader
url='https://en.wikipedia.org/wiki/Cricket'

loader=PyPDFLoader(r'C:\Users\SANSKAR\Desktop\LangChain\TextSplitter\Freelancer Annotation Onboarding Guide_FINAL.pdf')
text=loader.load()
final_text=text[0].page_content

splitter=CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    separator=''
)
result=splitter.split_text(final_text)
print(len(result))