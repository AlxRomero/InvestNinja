import os
import tempfile

import streamlit as st
from langchain.llms import VertexAI, OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma


st.title('InvestNinja')
st.subheader('Load your patent PDF, ask questions, and receive answers directly from the document.')

st.subheader('Upload your pdf')
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))

while uploaded_file is None:
    continue

if uploaded_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)

loader = PyPDFLoader(temp_file_path)
pages = loader.load_and_split()

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

store = Chroma.from_documents(pages, embeddings, collection_name='Pdf')

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

vectorstore_info = VectorStoreInfo(
    name="Pdf",
    description=" A pdf file to answer your questions",
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

prompt = st.text_input('Input your prompt here')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt) 
        st.write(search[0][0].page_content)
