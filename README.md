📄 AI-Powered PDF Query System

An advanced AI-based system that allows users to upload PDFs and ask questions based on their content.
Using Pinecone for vector storage and OpenAI embeddings, this system efficiently retrieves and answers queries with relevant document references.

📌 Features

✔️ Upload and process multiple PDFs

✔️ Uses OpenAI embeddings for document understanding

✔️ Stores and retrieves data using Pinecone vector database

✔️ Smart Q&A system with detailed answers

✔️ Provides document references for transparency

✔️ User-friendly Streamlit web interface

⚙️ How It Works

1️⃣ Upload PDFs – Users upload documents for processing.

2️⃣ Text Extraction – Extracts content using PyPDFLoader.

3️⃣ Chunking & Embedding – Splits text into sections and converts them into embeddings using OpenAI's model.

4️⃣ Storage in Pinecone – The system indexes the document embeddings in Pinecone for fast retrieval.

5️⃣ Ask Questions – Users submit a query, and the system searches the most relevant document sections.

6️⃣ Retrieve & Answer – Uses GPT-3.5 to generate a response based on retrieved documents.

🔧 Installation & Setup
# Create a virtual environment (optional but recommended)

python -m venv venv

# Activate the virtual environment

venv\Scripts\activate  

# Install dependencies

pip install -r requirements.txt
