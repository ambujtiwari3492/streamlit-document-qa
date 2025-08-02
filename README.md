Streamlit Document QA App
Project Overview
This is a Streamlit-based application designed to help developers and analysts get accurate, context-aware answers from large PDF documents, such as High-Level Design (HLD) documents. It addresses the common challenge of retrieving precise information from long, technical PDFs, where general-purpose AI tools may fall short.

Key Features
Accurate Answers: Provides specific answers to questions by verifying them against the content of the source PDF.

Chat History: Maintains conversational context, allowing for natural, follow-up questions.

User-Friendly Interface: Built with Streamlit for a simple and intuitive web experience.

Technology Stack
The application uses a powerful RAG (Retrieval-Augmented Generation) architecture, leveraging a combination of open-source tools:

Streamlit: For creating the interactive web interface.

LangChain: To orchestrate the document loading, splitting, and retrieval logic.

FAISS: A fast vector database for efficient storage and retrieval of document chunks.

Hugging Face: Utilized for a robust sentence transformer model to generate document embeddings.

Chunking: The PDF documents are broken down into smaller, manageable chunks to improve the accuracy of the semantic search.

Google Gemini API: The generative model used to formulate answers based on the retrieved context.

Getting Started
Prerequisites
Python 3.8+

Git

Installation
Clone the repository:

git clone https://github.com/ambujtiwari3492/streamlit-document-qa.git
cd streamlit-document-qa



Create and activate a virtual environment:

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the dependencies:

pip install -r requirements.txt

Configuration
Get a Google Gemini API Key:
Go to the Google AI Studio to get your API key.

Create a .env file:
In the root directory of the project, create a file named .env and add your API key to it:

GOOGLE_API_KEY="your_api_key_here"

Running the App
Once all the steps are complete, you can run the Streamlit application:

streamlit run app2.py ---Gemini

streamlit run app3.py --- with chat history

The app will open in your web browser, where you can upload your PDF and start asking questions.