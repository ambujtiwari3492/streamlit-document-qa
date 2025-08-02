import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Hugging Face API Key
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_API_KEY:
    st.error("Missing HuggingFace API key. Please set it in your .env file.")
    st.stop()


# Streamlit file uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if uploaded_file is not None:
    with st.spinner("Processing PDF and building index..."):
        # Save uploaded PDF to a temp file
        temp_pdf_path = os.path.join("temp_uploaded.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        # Load PDF and build FAISS index
        documents = PyPDFLoader(temp_pdf_path).load()
        db = FAISS.from_documents(documents, embeddings)
        retriever = db.as_retriever()
else:
    st.info("Please upload a PDF to begin.")
    retriever = None

# Prompt template
prompt_template = """
You are an AI assistant. Use the following pieces of context to answer the question.
If you don't know the answer, just say you don't know. Be concise.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# Streamlit UI
st.title("Document QA App")

question = st.text_input("Ask a question about your documents:")


def generate_answer(prompt_text, api_token):
    # This is the most reliable way to use the Hugging Face Inference API.
    # We'll use a model known to be available on the Inference Providers service.
    model = "meta-llama/Llama-2-7b-chat-hf"  # Llama-2 is a good choice for this type of task
    
    try:
        client = InferenceClient(model=model, token=api_token)
        response = client.text_generation(
            prompt_text, 
            max_new_tokens=150, 
            return_full_text=False
        )
        return response
    except Exception as e:
        st.error(f"Error generating response from Hugging Face API: {e}")
        return None


if question and retriever is not None:
    with st.spinner("Generating answer..."):
        # Retrieve context from FAISS index
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build prompt
        prompt_text = prompt_template.format(context=context, question=question)
        
        # Generate answer using the new function
        answer = generate_answer(prompt_text, HF_API_KEY)
        
        if answer is not None:
            st.write("Answer:", answer)