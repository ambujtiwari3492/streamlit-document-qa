import os
import warnings
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the Gemini API key from the .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Missing Google API key. Please set it in your .env file.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit file uploader for PDF
st.title("Document QA App")
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if uploaded_file is not None:
    with st.spinner("Processing PDF and building index..."):
        try:
            # Use tempfile to create a temporary file that is automatically cleaned up
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Load PDF and build FAISS index from the temporary path
            documents = PyPDFLoader(tmp_path).load()
            db = FAISS.from_documents(documents, embeddings)
            retriever = db.as_retriever()

            # Clean up the temporary file
            os.remove(tmp_path)
            st.success("PDF processed and index built successfully!")

        except Exception as e:
            st.error(f"An error occurred during PDF processing: {e}")
            retriever = None
else:
    st.info("Please upload a PDF to begin.")
    retriever = None

# Prompt template
# This template provides instructions to the model, ensuring it focuses on the provided context
prompt_template = """
You are an AI assistant. Use only the following pieces of context to answer the question.
Your answers should be based strictly on the information provided in the context.
If the answer is not in the context, politely state that you don't have enough information.
Answer concisely and do not add any external knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

question = st.text_input("Ask a question about your documents:")

def generate_answer_with_gemini(prompt_text):
    # We are now using a current and stable model, `gemini-2.5-flash`, for text generation.
    # The 'models/' prefix is necessary for the API to locate the model.
    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini API: {e}")
        return None

if question and retriever is not None:
    with st.spinner("Generating answer..."):
        try:
            # Retrieve context from FAISS index
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Build prompt
            prompt_text = prompt_template.format(context=context, question=question)
            
            # Generate answer using the new function
            answer = generate_answer_with_gemini(prompt_text)
            
            if answer is not None:
                st.write("Answer:", answer)
        except Exception as e:
            st.error(f"An error occurred while retrieving information: {e}")