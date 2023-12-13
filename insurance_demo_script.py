import os
import openai
import pickle
import PyPDF2
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.callbacks import get_openai_callback

openai.api_key = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.title("Genius Health - PDF parser with LLM")
    st.markdown('''
       ## About
       This app is an LLM generic pdf parser 
    ''')
    add_vertical_space(5)

def main():
    st.header("Chat with PDF")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf:
        pdf_reader = PyPDF2.PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        if not text:
            print("Failed to parse text")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write("embeddings loaded from disk")
        else:
            # embeddings
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            # st.write("embeddings computation completed")

        # Accept user questions/query
        query = st.text_input("Ask any question about your PDF file:")
        if query: 
            docs = VectorStore.similarity_search(query=query, k=5)
            llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)    
            st.write(response)

main()
