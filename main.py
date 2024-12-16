# import streamlit as st
# import os
# import hashlib
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import tempfile
# import shutil

# # Load environment variables
# load_dotenv()

# # Access the API key from Streamlit secrets
# api_key = st.secrets['gapi']

# # Configure Google Generative AI with the API key
# genai.configure(api_key=api_key)

# # Initialize session state if not already initialized
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Define folder for local PDFs
# PDF_FOLDER = "./HTMs"  # Path to the local folder containing the PDFs

# # Function to get the file hash (to track changes)
# def generate_file_hash(file_path):
#     hash_sha256 = hashlib.sha256()
#     with open(file_path, "rb") as f:
#         while chunk := f.read(8192):  # Read file in chunks
#             hash_sha256.update(chunk)
#     return hash_sha256.hexdigest()

# # Get text from PDFs in the local folder
# def get_pdf_text(pdf_docs):
#     text = ""
#     metadata = []
#     for pdf_doc in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf_doc)
#             for page_num, page in enumerate(pdf_reader.pages):
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text
#                     metadata.append({"source": pdf_doc, "page_number": page_num})
#         except Exception as e:
#             st.error(f"Error reading PDF: {e}")
#     return text, metadata

# def get_text_chunks(text, metadata):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
    
#     chunk_metadata = []
#     for i, chunk in enumerate(chunks):
#         chunk_metadata.append(metadata[i % len(metadata)])  # Ensure each chunk has corresponding metadata
    
#     return chunks, chunk_metadata

# # Save FAISS index to local storage
# def get_vector_store(text_chunks, metadata):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     if len(text_chunks) != len(metadata):
#         raise ValueError(f"Texts and metadata must be the same length.")
    
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details and you answer the question based on the context provided and not hallucinate.
#     If the answer is not found in the provided context or if the context does not have enough information for you to answer the question, inform that the relevant context was not available in the given context and add 'it seems the relevant information is not available in uploaded PDFs to answer your question, please ask a question related to the contents of the PDFs uploaded.'  
#     \n\nContext:\n {context}?\nQuestion: \n{question}\nAnswer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # User input processing
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

#     st.chat_message("user").markdown(user_question)
#     st.session_state.messages.append({"role": "user", "content": user_question})

#     assistant_response = response["output_text"]
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)
#     st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# # Check if the vector store needs updating
# def update_vector_store():
#     # Get list of PDFs in the HTMs folder
#     pdf_files = [os.path.join(PDF_FOLDER, file) for file in os.listdir(PDF_FOLDER) if file.endswith('.pdf')]

#     if os.path.exists("faiss_index"):
#         # Get stored file hashes (could be in a file or memory)
#         stored_hashes = set()
#         if os.path.exists("stored_hashes.txt"):
#             with open("stored_hashes.txt", "r") as f:
#                 stored_hashes = set(line.strip() for line in f)

#         # Get current file hashes
#         current_hashes = set(generate_file_hash(file) for file in pdf_files)

#         # Check if there are any new or modified files
#         if current_hashes != stored_hashes:
#             st.warning("New or updated files found. Rebuilding the vector store.")
#             raw_text = ""
#             metadata = []
#             for pdf_file in pdf_files:
#                 raw_text_part, metadata_part = get_pdf_text([pdf_file])
#                 raw_text += raw_text_part
#                 metadata.extend(metadata_part)

#             # Split text into chunks
#             text_chunks, chunk_metadata = get_text_chunks(raw_text, metadata)

#             # Save the new vector store
#             get_vector_store(text_chunks, chunk_metadata)

#             # Update stored hashes
#             with open("stored_hashes.txt", "w") as f:
#                 for hash in current_hashes:
#                     f.write(f"{hash}\n")
#         else:
#             st.success("No new updates detected.")
#     else:
#         st.warning("No existing vector store found. Rebuilding the vector store.")
#         raw_text = ""
#         metadata = []
#         for pdf_file in pdf_files:
#             raw_text_part, metadata_part = get_pdf_text([pdf_file])
#             raw_text += raw_text_part
#             metadata.extend(metadata_part)

#         # Split text into chunks
#         text_chunks, chunk_metadata = get_text_chunks(raw_text, metadata)

#         # Save the new vector store
#         get_vector_store(text_chunks, chunk_metadata)

#         # Store hashes to track future changes
#         current_hashes = set(generate_file_hash(file) for file in pdf_files)
#         with open("stored_hashes.txt", "w") as f:
#             for hash in current_hashes:
#                 f.write(f"{hash}\n")

# # Main Streamlit app
# def main():
#     st.set_page_config("Chat with PDF")
#     st.header("Chat with PDFs using Gemini")

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     user_question = st.chat_input("Ask a question to answer from the PDF files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Manage PDFs from Local Folder")
#         if st.button("Update & Process Files"):
#             with st.spinner("Processing..."):
#                 update_vector_store()
#                 st.success("Vector store updated and ready to answer questions.")

# if __name__ == "__main__":
#     main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access the API key from Streamlit secrets
api_key = st.secrets['gapi']

# Configure Google Generative AI with the API key
genai.configure(api_key=api_key)

# Initialize session state if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []
# Read PDFs loaded and get all text from all pages in one text (context)
def get_pdf_text(pdf_docs):
    text = ""
    metadata = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    # Store metadata per page
                    metadata.append({"source": pdf.name, "page_number": page_num})
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text, metadata


def get_text_chunks(text, metadata):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    # Ensure each chunk has corresponding metadata
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_metadata.append(metadata[i % len(metadata)])  # Cycle metadata if necessary (but ideally should align correctly)
    
    return chunks, chunk_metadata


# FAISS
def get_vector_store(text_chunks, metadata):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Ensure text chunks and metadata are of the same length
    if len(text_chunks) != len(metadata):
        raise ValueError(f"Texts and metadata must be of the same length. Got {len(text_chunks)} texts and {len(metadata)} metadata.")
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details and you answer the question based on the context provided and not hallucinate.
    If the answer is not found in the provided context or if the context does not have enough information for you to answer the question, inform that the relevant context was not available in the given context and add 'it seems the relevant information is not available in uploaded PDFs to answer your question, please ask a question related to the contents of the PDFs uploaded.'
    
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Display user question
    st.chat_message("user").markdown(user_question)
    
    # Add user question to the chat history
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Generate assistant response
    assistant_response = response["output_text"]
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    
    # Add assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})


def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDFs using Gemini")

    # Display the conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Use st.chat_input to capture user input
    user_question = st.chat_input("Ask a question to answer from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload the PDF files", accept_multiple_files=True)
        
        # Sidebar information about the chatbot's purpose
        st.sidebar.markdown(
            """
            **Purpose:** 
            This chatbot extracts information from uploaded PDFs and answers questions based only on the provided context. 
            Ensure your question aligns with the content of the PDFs.
            """
        )
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text, metadata = get_pdf_text(pdf_docs)  # Get both text and metadata
                text_chunks, chunk_metadata = get_text_chunks(raw_text, metadata)  # Pass both text and metadata
                get_vector_store(text_chunks, chunk_metadata)  # Pass chunk_metadata along with text_chunks
                st.success("Done, now it is ready to answer your question!")


if __name__ == "__main__":
    main()


# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

# import os
# from dotenv import load_dotenv

# # Access the API key from Streamlit secrets
# api_key = st.secrets['gapi']

# # Configure Google Generative AI with the API key
# genai.configure(api_key=api_key)

# # Initialize session state if not already initialized
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Read PDFs loaded and get all text from all pages in one text (context)
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks


# # FAISS
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details and you answer the question based on the context provided and not hallucinate.
#     If the answer is not found in the provided context or if the context does not have enough information for you to answer the question, inform that the relevant context was not available in the given context and add 'it seems the relevant information is not available in uploaded PDFs to answer your question, please ask a question related to the contents of the PDFs uploaded.'
    
#     \n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n
#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain


# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )

#     # Display user question
#     st.chat_message("user").markdown(user_question)
    
#     # Add user question to the chat history
#     st.session_state.messages.append({"role": "user", "content": user_question})

#     # Generate assistant response
#     assistant_response = response["output_text"]
    
#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)
    
#     # Add assistant response to the chat history
#     st.session_state.messages.append({"role": "assistant", "content": assistant_response})


# def main():
#     st.set_page_config("Chat with PDF")
#     st.header("Chat with PDFs using Gemini")

#     # Display the conversation history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Use st.chat_input to capture user input
#     user_question = st.chat_input("Ask a question to answer from the PDF files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Upload your PDFs")
#         pdf_docs = st.file_uploader("Upload the PDF files", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done, now it is ready to answer your question!")

# print("ALL GOOD")


# if __name__ == "__main__":
#     main()

# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
# import os
# from dotenv import load_dotenv

# # Access the API key from Streamlit secrets
# api_key = st.secrets['gapi']

# # Configure Google Generative AI with the API key
# genai.configure(api_key=api_key)

# # Initialize session state if not already initialized
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Thread-safe FAISS Handler
# class SafeFAISS:
#     def __init__(self):
#         self.lock = Lock()
#         self.vector_store = None

#     def load_index(self, embeddings):
#         with self.lock:
#             try:
#                 self.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#             except:
#                 self.vector_store = FAISS.from_texts([], embedding=embeddings)

#     def query(self, user_question, k=5):
#         with self.lock:
#             return self.vector_store.similarity_search(user_question, k=k)

#     def add_texts(self, chunks, metadata, embeddings):
#         with self.lock:
#             self.vector_store.add_texts(chunks, metadata, embedding=embeddings)

#     def save_index(self):
#         with self.lock:
#             self.vector_store.save_local("faiss_index")

# # Thread-safe FAISS handler instance
# faiss_handler = SafeFAISS()

# # PDF Text Extraction
# # Extract text from a single PDF
# def extract_text_from_pdf(pdf):
#     pdf_reader = PdfReader(pdf)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# # Combine text extraction from multiple PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         text += extract_text_from_pdf(pdf)
#     return text

# # Text Chunking
# def get_text_chunks_parallel(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Conversational Chain Setup
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details and you answer the question based on the context provided and not hallucinate.
#     If the answer is not found in the provided context or if the context does not have enough information for you to answer the question, inform that the relevant context was not available in the given context and add 'it seems the relevant information is not available in uploaded PDFs to answer your question, please ask a question related to the contents of the PDFs uploaded.'
    
#     \n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Update the FAISS vector store in parallel
# def update_vector_store_parallel(text_chunks, pdf_names):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     faiss_handler.load_index(embeddings)

#     with ThreadPoolExecutor() as executor:
#         tasks = [
#             executor.submit(lambda chunk: faiss_handler.add_texts([chunk], [{"source": pdf_names[0]}], embeddings), chunk)
#             for chunk in text_chunks
#         ]

#         for task in tasks:
#             task.result()  # Ensure all tasks complete

#     # Save the updated FAISS index
#     faiss_handler.save_index()

# # Query User Input and Generate Response
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # Ensure FAISS index is loaded
#     if faiss_handler.vector_store is None:
#         faiss_handler.load_index(embeddings)

#     # Perform a thread-safe query
#     docs = faiss_handler.query(user_question, k=5)

#     # Combine retrieved chunks with metadata
#     retrieved_docs = [
#         f"From {doc.metadata.get('source', 'Unknown Source')}: {doc.page_content}"
#         for doc in docs
#     ]
#     combined_context = "\n".join(retrieved_docs)

#     # Generate a response using the conversational chain
#     chain = get_conversational_chain()
#     response = chain({"context": combined_context, "question": user_question}, return_only_outputs=True)

#     # Display and store user input and assistant response
#     st.chat_message("user").markdown(user_question)
#     st.session_state.messages.append({"role": "user", "content": user_question})

#     assistant_response = response["output_text"]
#     with st.chat_message("assistant"):
#         st.markdown(assistant_response)

#     st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# # Main Application
# def main():
#     st.set_page_config("Chat with PDF")
#     st.header("Chat with PDFs using Gemini")

#     # Display the conversation history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Use st.chat_input to capture user input
#     user_question = st.chat_input("Ask a question to answer from the PDF files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Upload your PDFs")
#         pdf_docs = st.file_uploader("Upload the PDF files", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 # Process new PDFs
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks_parallel(raw_text)

#                 # Update FAISS index in parallel
#                 pdf_names = [pdf.name for pdf in pdf_docs]
#                 update_vector_store_parallel(text_chunks, pdf_names)

#                 st.success("Done, now it is ready to answer your question!")

# print("ALL GOOD")

# if __name__ == "__main__":
#     main()

