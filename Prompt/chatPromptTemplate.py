from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
# chat_template=ChatPromptTemplate([
#     SystemMessage(content='You are a helpful {domain} assistant.'),
#     HumanMessage(content='Explain {concept} in simple terms?')
# ])
chat_template=ChatPromptTemplate([
    ('system','you are a helpful {domain} assistant.'),
    ('human','Explain { concept} in simple terms?')
])
prompt= chat_template.invoke({'domain':'cricket','concept':'noball'})

print(prompt)