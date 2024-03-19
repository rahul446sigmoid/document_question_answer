import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import random
import string
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from  css import render_navbar
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from streamlit_option_menu import option_menu
st.set_page_config(
        page_title="Sigmoid Style",
        layout="wide",
    )

render_navbar()


# if selected == "DocAI":
with st.container(height=100):  
    openai_api_key = st.text_input("Please Enter Your OpenAI Key",type='password')
def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Create a vectorstore from documents
        db = FAISS.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        return qa.run(query_text),documents,texts,embeddings
def pdf_chat(uploaded_file,query):

    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = uploaded_file.name[:-4]
        
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        #Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            

        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            #st.write(docs)
            
            #openai rank lnv process
            llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
            chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                
    return response,text,chunks,embeddings
def main():
    # File upload
    with st.container(height=200):
        uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf'])
    selected = option_menu(None, ['DOC-Chat','Working Mechanism'], 
    icons=['person','gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#eee"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#F15050"},
    }
    )
    if selected=='DOC-Chat':
        def chat_actions():
            st.session_state["chat_history"].append(
                {"role": "user", "content": st.session_state["chat_input"]},
            )
            if uploaded_file.type == 'text/plain':
                response = generate_response(uploaded_file, openai_api_key, st.session_state["chat_input"])[0]
                
            elif uploaded_file.type == 'application/pdf':
                # Handle plain text file
                response = pdf_chat(uploaded_file,st.session_state["chat_input"])[0]
                
            else:
                response = "Unsupported file format. Please upload a PDF or text file."
        
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": response,
                },  # This can be replaced with your chat response logic
            )


        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []


        st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")

        for i in st.session_state["chat_history"]:
            with st.chat_message(name=i["role"]):
                st.write(i["content"])
    if selected =='Working Mechanism':
       
        query= st.text_input("Enter Your Question Here..")
        if query:
            if uploaded_file.type == 'text/plain':
                re = generate_response(uploaded_file, openai_api_key,query )
                st.markdown("## The file that you have uploaded")
                st.write(re[1])
                st.markdown("## Smaller Chunks")
                st.write(re[2])
                st.markdown("## Embedding")
                st.write(re[3].embed_query(query)[:15])
                st.write(re[1])   
            elif uploaded_file.type == 'application/pdf':
                    # Handle plain text file
                re= pdf_chat(uploaded_file,query)
                st.markdown("## The file that you have uploaded")
                st.write(re[1])
                st.markdown("## Smaller Chunks")
                st.write(re[2])
                st.markdown("## Embedding")
                st.write(re[3].embed_query(query)[:15])
   
        





if __name__ == "__main__":
    main()
