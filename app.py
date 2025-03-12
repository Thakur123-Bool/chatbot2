import os
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

app = Flask(__name__, static_folder='static', template_folder='templates')

# Ensure directories for vectorstore and uploads exist
if not os.path.exists('./chroma_db'):
    os.makedirs('./chroma_db')

if not os.path.exists('./uploads'):
    os.makedirs('./uploads')

# Global variable to store processed PDFs
processed_pdfs = None

# Serve the main HTML page
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

# Function to process PDFs and extract text with metadata
def process_pdf(pdf_files):
    if not pdf_files:
        return None  # Return None if no PDFs are uploaded

    all_chunks = []
    try:
        for pdf_file in pdf_files:
            file_path = os.path.join('./uploads', pdf_file.filename)  # Save the file path
            pdf_file.save(file_path)  # Save the uploaded PDF file
            
            loader = PyMuPDFLoader(file_path)
            data = loader.load()

            # Extract metadata (page numbers and file paths)
            for doc in data:
                doc.metadata['page_number'] = doc.metadata.get('page', 0) + 1  # Page numbers start from 1
                doc.metadata['file_path'] = file_path  # Store the file path

            # Split text into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            all_chunks.extend(chunks)

        # Create embeddings and store in Chroma
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        
        # Create vectorstore and retriever
        vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        return vectorstore, retriever
    except Exception as e:
        print(f"Error during PDF processing: {str(e)}")
        return None

# Function to search for exact matches in the PDFs
def search_in_pdfs(question, retriever):
    if not question.strip():
        return None, None, None  # Return None if the question is empty

    try:
        # Retrieve relevant chunks
        retrieved_docs = retriever.invoke(question)

        # Check if any relevant documents were found
        if not retrieved_docs:
            return None, None, None

        # Extract the most relevant document
        most_relevant_doc = retrieved_docs[0]
        page_number = most_relevant_doc.metadata['page_number']
        file_path = most_relevant_doc.metadata['file_path']
        content = most_relevant_doc.page_content

        return content, page_number, file_path
    except Exception as e:
        print(f"Error during document search: {str(e)}")
        return None, None, None

# API route to handle user questions
@app.route('/ask', methods=['POST'])
def ask_question():
    global processed_pdfs  # Declare that we're using the global variable

    # Check if the request contains PDFs
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No PDF files uploaded"}), 400

    pdf_files = request.files.getlist('pdf_files')
    question = request.form.get('question')

    # Check if the question is empty
    if not question.strip():
        return jsonify({"error": "Please provide a question"}), 400

    # Process PDFs if not already processed
    if processed_pdfs is None:
        processed_pdfs = process_pdf(pdf_files)

    if processed_pdfs is None:
        return jsonify({"error": "No PDFs were processed. Please upload valid PDF files."}), 400

    vectorstore, retriever = processed_pdfs

    # Search for the question in the PDFs
    content, page_number, file_path = search_in_pdfs(question, retriever)

    if not content:
        return jsonify({"error": "The question is irrelevant to the uploaded documents. Please ask a question related to the documents."}), 404

    response = {
        "answer": content,
        "page_number": page_number,
        "file_path": file_path
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)


