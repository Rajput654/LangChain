from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os 
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)
model =ChatHuggingFace(llm=llm)

st.header("Personal Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input=st.chat_input("Ask me ....")

template=PromptTemplate.from_template(
      "You are a helpful assistant.\n"
    "Chat History:\n{chat_history}\n"
    "User: {user_input}\n"
    "Assistant:"
)

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})

    chat_history="\n".join(
        [f'{m["role"]}: m{["content"]}' for m in st.session_state.messages]
    )

    chain= template|model
    result =chain.invoke({
        "user_input":user_input,
        "chat_history":chat_history
    })

    response=result.content

    st.session_state.messages.append({"role":"assistant","content":response})

    with st.chat_message("assistant"):
        st.write(response)