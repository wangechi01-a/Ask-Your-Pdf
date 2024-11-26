import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import base64

# Load environment variables
load_dotenv()

# Google Generative AI API Key
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text().replace("\n", " ")  
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant tasked with answering questions based on the provided context from a PDF document. Use the following rules while answering:

    1. **Context-Based Answers**: Always refer to the provided context to answer the question.
    2. **Handling Partial Context**: If the context partially answers the question, provide as much detail as possible and clearly indicate that additional information is not available.
    3. **No Context Match**: If the answer is not in the context, respond with: "The answer is not available in the provided context."
    4. **Avoid Guesswork**: Do not make up information or provide incorrect answers.
    5. **Explain Reasoning**: Clearly explain how the provided context was used to derive your answer.
    Context: {context}
    Question: {question}
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)  # Use Gemini Pro model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

@st.cache_data
def displayPDF(file):
    # Read the uploaded file as bytes
    file_bytes = file.read()
    
    # Convert the file bytes to base64 for embedding in HTML iframe
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():
    # Custom CSS to style the title
    st.markdown("""
    <style>
    .title {
        color: #1E90FF;  /* Blue color */
        font-size: 36px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Ask Your PDF 📄</div>', unsafe_allow_html=True)

    user_question = st.text_input("Input your Query Here!")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # Display the uploaded PDFs in the sidebar
        if pdf_docs:
            for pdf in pdf_docs:
                displayPDF(pdf)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
