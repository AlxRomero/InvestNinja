import os
import tempfile

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'application_default_credentials.json'

from google.cloud import aiplatform

aiplatform.init(
    project='investninja'
)

import streamlit as st
from langchain.llms import VertexAI, OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


st.title('InvestNinja 🥷')
st.subheader('Load your patent PDF, ask questions, and receive investment insights directly sourced from the document.')

st.subheader('Upload your pdf')
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))

if uploaded_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # st.write("Full path of the uploaded file:", temp_file_path)

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    llm = OpenAI(temperature=0.1, verbose=True, request_timeout=120)
    #llm = VertexAI(temperature=0.1, verbose=True, request_timeout=120)
    embeddings = OpenAIEmbeddings()

    store = Chroma.from_documents(pages, embeddings, collection_name='patent-data')

    vectorstore_info = VectorStoreInfo(
        name="pdf",
        description="A pdf file that contains patent text data",
        vectorstore=store
    )

    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    prompt = "What is summary of the pdf?"

    response = agent_executor.run(prompt)
    st.write(response)

    input_from_user = st.text_input('Ask you question here!')

    if input_from_user:
        response = agent_executor.run(input_from_user)
        st.write(response)

        with st.expander('Document Similarity Search'):
            search = store.similarity_search_with_score(input_from_user)
            st.write(search[0][0].page_content)
