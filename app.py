import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import pypdf

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("API keys are missing! Please set them in your environment variables.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Index Name
index_name = "langchainvector"

# Check if Index Exists, Otherwise Create It
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Ensure "uploads" directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to Load PDF Files After Saving Locally
def load_pdfs(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join(UPLOAD_FOLDER, pdf_file.name)
        # Save file locally
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    return docs

# Function to Split Documents into Chunks
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Function to Process PDFs and Store in Pinecone
def process_and_store_pdfs(pdf_files):
    documents = load_pdfs(pdf_files)
    chunks = split_docs(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=index_name
    )
    return vector_store

# Function to Answer Queries with a Single Reference
def answer_question_with_references(question):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Load Pinecone vector store from the existing index
    vector_store = PineconeVectorStore.from_existing_index(index_name, embeddings)
    # Set retriever to fetch only one document (top result)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    # Define the Chat model
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    # Define a prompt template for detailed responses
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an AI assistant answering questions based on the provided documents.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer with detailed explanation and include reference."
        ),
    )
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={"prompt": prompt}
    )
    # Run query by passing the query as a dictionary
    response = qa_chain({"query": question})
    
    # Retrieve only the top relevant document for reference
    retrieved_docs = retriever.get_relevant_documents(question)
    reference = ""
    if retrieved_docs:
        doc = retrieved_docs[0]
        metadata = doc.metadata
        source = metadata.get("source", "Unknown Source")
        page = metadata.get("page", "Unknown Page")
        reference = f"{source} (Page {page})"
    return response, [reference] if reference else []

# Configure Streamlit page layout and custom CSS styling
st.set_page_config(page_title="PDF Query System", layout="wide")
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {background-color: #f0f2f6;}
    /* Sidebar styling */
    .sidebar .sidebar-content {background-color: #ffffff; padding: 20px;}
    /* Button styling */
    .upload-btn, .search-btn {
        padding: 10px 20px; border: none; border-radius: 5px; font-size: 1rem; cursor: pointer;
    }
    .upload-btn {background-color: #007bff; color: white;}
    .upload-btn:hover {background-color: #0056b3;}
    .search-btn {background-color: #28a745; color: white;}
    .search-btn:hover {background-color: #1e7e34;}
    </style>
    """, unsafe_allow_html=True
)

# UI: Sidebar for PDF uploads
st.sidebar.title("Upload PDFs")
pdf_files = st.sidebar.file_uploader("Select PDF files", accept_multiple_files=True, type=["pdf"])
if st.sidebar.button("Process PDFs"):
    if pdf_files:
        process_and_store_pdfs(pdf_files)
        st.sidebar.success("PDFs processed ")
    else:
        st.sidebar.warning("Please upload at least one PDF file.")

# Main UI: Q&A interface
st.title("📄 PDF Query System")
st.markdown("### Ask a Question")
question = st.text_input("Enter your question here:")
if st.button("Get Answer"):
    if question:
        answer, references = answer_question_with_references(question)
        st.markdown("#### Answer:")
        st.write(answer)
        if references:
            st.markdown("#### Reference:")
            for ref in references:
                st.write(f"- {ref}")
    else:
        st.warning("Please enter a question.")
