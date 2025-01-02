import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from google import genai
from google.genai import types
import textwrap
from dataclasses import dataclass
from PIL import Image
import time
from ratelimit import limits, sleep_and_retry
import fitz
import io
from dotenv import load_dotenv
import streamlit as st

@dataclass
class Config:
    """Configuration class for the application"""
    MODEL_NAME: str = "gemini-2.0-flash-exp"  # Updated to match Google's example
    TEXT_EMBEDDING_MODEL_ID: str = "text-embedding-004"  # Correct embedding model name
    DPI: int = 300  # Resolution for PDF to image conversion

class PDFProcessor:
    """Handles PDF processing using PyMuPDF and Gemini's vision capabilities"""
    
    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int = Config.DPI) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert PyMuPDF pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        pdf_document.close()
        return images

    @sleep_and_retry
    @limits(calls=30, period=60)
    def create_embeddings(self, data: str):
        """Create embeddings with rate limiting - exactly as in Google's example"""
        time.sleep(1)
        return self.client.models.embed_content(
            model=Config.TEXT_EMBEDDING_MODEL_ID,
            contents=data,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )

    def find_best_passage(self, query: str, dataframe: pd.DataFrame) -> dict:
        """Find the most relevant passage for a query"""
        try:
            query_embedding = self.client.models.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_ID,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            
            dot_products = np.dot(np.stack(dataframe['Embeddings']), 
                                query_embedding.embeddings[0].values)
            idx = np.argmax(dot_products)
            content = dataframe.iloc[idx]['Original Content']
            return {
                'page': content['page_number'],
                'content': content['content']
            }
        except Exception as e:
            print(f"Error finding best passage: {e}")
            return {'page': 0, 'content': ''}

def make_answer_prompt(self, query: str, passage: dict) -> str:
        """Create prompt for answering questions"""
        escaped = passage['content'].replace("'", "").replace('"', "").replace("\n", " ")
        return textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
                                 You are answering questions about a research paper. 
                                 Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
                                 However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
                                 strike a friendly and conversational tone. 
                                 If the passage is irrelevant to the answer, you may ignore it.
                                 
                                 QUESTION: '{query}'
                                 PASSAGE: '{passage}'
                                 
                                 ANSWER:
                              """).format(query=query, passage=escaped)


class RAGApplication:
    """Main RAG application class"""
    
    def __init__(self, api_key: str):
        self.gemini_client = GeminiClient(api_key)
        self.data_df = None
        
    def process_pdf(self, pdf_path: str):
        """Process PDF using Gemini's vision capabilities"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Convert PDF pages to images
        images = PDFProcessor.pdf_to_images(pdf_path)
        
        # Analyze each page
        page_contents = []
        page_analyses = []
        
        st.write("Analyzing PDF pages...")
        for i, image in enumerate(tqdm(images)):
            content = self.gemini_client.analyze_page(image)
            if content:
                # Store both the analysis and the content
                page_contents.append({
                    'page_number': i+1,
                    'content': content
                })
                page_analyses.append(content)
        
        if not page_analyses:
            raise ValueError("No content could be extracted from the PDF")
            
        # Create dataframe
        self.data_df = pd.DataFrame({
            'Original Content': page_contents,
            'Analysis': page_analyses
        })
        
        # Generate embeddings
        st.write("\nGenerating embeddings...")
        embeddings = []
        try:
            for text in tqdm(self.data_df['Analysis']):
                embeddings.append(self.gemini_client.create_embeddings(text))
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            time.sleep(10)
            
        _embeddings = []
        for embedding in embeddings:
            _embeddings.append(embedding.embeddings[0].values)
            
        self.data_df['Embeddings'] = _embeddings


def answer_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        """Answer a list of questions using the processed data"""
        if self.data_df is None:
            raise ValueError("Please process a PDF first using process_pdf()")
            
        answers = []
        for question in questions:
            try:
                passage = self.gemini_client.find_best_passage(question, self.data_df)
                prompt = self.gemini_client.make_answer_prompt(question, passage)
                response = self.gemini_client.client.models.generate_content(
                    model=Config.MODEL_NAME,
                    contents=prompt
                )
                answers.append({
                    'question': question,
                    'answer': response.text,
                    'source': f"Page {passage['page']}\nContent: {passage['content']}"
                })
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                answers.append({
                    'question': question,
                    'answer': f"Error generating answer: {str(e)}",
                    'source': "Error"
                })
            
        return answers

def main():
    import streamlit as st

    # Page title
    st.set_page_config(page_title='🦜🔗 Ask the Doc App')
    st.title('🦜🔗 Ask the Doc App')

    # Hardcoded API key
    api_key = st.secrets['gapi']

    if not api_key:
        raise ValueError("API key is missing.")

    # Test the API key
    try:
        test_client = genai.Client(api_key=api_key)
        test_response = test_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Hello, this is a test message."
        )
        st.write("API key is working!", test_response.text)
    except Exception as e:
        st.write(f"API test failed: {e}")
        raise ValueError("Invalid API key.")

    # Form
    with st.form(key="stimy_form"):
        pdf_path = st.file_uploader("Upload a PDF file", type=["pdf"])
        questions = st.text_input('Enter your question:', placeholder='Please provide a short summary.')
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and pdf_path and questions:
        try:
            # Save the uploaded PDF to a temp file
            temp_pdf_path = f"temp_{pdf_path.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_path.getbuffer())

            # Initialize application
            app = RAGApplication(api_key)

            # Process PDF and answer questions
            st.write(f"Processing PDF: {pdf_path.name}")
            with st.spinner("Thinking..."):
                app.process_pdf(temp_pdf_path)
                answers = app.answer_questions(questions)

            # Display answers
            for result in answers:
                st.write(f"Question: {result['question']}")
                st.write(f"Answer: {result['answer']}")
                st.write(f"Source: {result['source']}")
                st.write("-" * 80)
        except Exception as e:
            st.write(f"An error occurred: {e}")

if __name__ == "__main__":
    main()






