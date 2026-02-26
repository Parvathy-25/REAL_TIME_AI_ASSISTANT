import streamlit as st

from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


#page config

st.set_page_config(page_title="Ollama RAG Chatbot",layout="centered")

st.title("Real-time AI Assistant")
st.caption("Powered by Ollama + DuckDuckGo + LangChain")

#initialize LLM& tools

llm=OllamaLLM(model="llama3:8b")
search=DuckDuckGoSearchRun()

prompt=ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. You must answer the user's question
    based *only* on the following search results.
    If the search results are empty or do not contain the answer,
    say "I could not find any information on that."

    Search Results:
    {context}                                     

    Question:
    {question}
    """
)

#chain creation 
chain=(
    RunnablePassthrough.assign(
        context=lambda x: search.run(x["question"])
        )
        | prompt
        | llm
)



if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_query=st.chat_input("Ask me anything...")

if user_query:

    st.session_state.messages.append(
        {"role":"user","content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response=chain.invoke({"question":user_query})
            st.markdown(response)

    st.session_state.messages.append(
        {"role":"assistant","content":response}

    )
    



