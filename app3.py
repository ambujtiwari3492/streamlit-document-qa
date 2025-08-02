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

# Streamlit app title
st.title("Document QA App with Chat History")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit file uploader for PDF
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
            st.session_state.retriever = db.as_retriever()

            # Clean up the temporary file
            os.remove(tmp_path)
            st.success("PDF processed and index built successfully!")

        except Exception as e:
            st.error(f"An error occurred during PDF processing: {e}")
            st.session_state.retriever = None
else:
    st.info("Please upload a PDF to begin.")
    st.session_state.retriever = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Prompt template, updated to include chat history
prompt_template = """
You are an AI assistant. Use the following chat history and context to answer the question.
Your answers should be based strictly on the information provided in the context.
If the answer is not in the context, politely state that you don't have enough information.
Answer concisely and do not add any external knowledge.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])

# Get user input using st.chat_input
question = st.chat_input("Ask a question about your documents:")

if question:
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Check if retriever is available
    if 'retriever' in st.session_state and st.session_state.retriever is not None:
        with st.spinner("Generating answer..."):
            try:
                # Retrieve context from FAISS index
                docs = st.session_state.retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Format the chat history into a string
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                
                # Build the full prompt with chat history
                prompt_text = prompt_template.format(chat_history=chat_history, context=context, question=question)
                
                # Generate answer using Gemini
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt_text)
                answer = response.text
                
                # Add assistant response to chat history and display
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.write(answer)
            except Exception as e:
                st.error(f"An error occurred while retrieving information: {e}")
    else:
        st.info("Please upload a PDF and wait for it to be processed before asking a question.")
