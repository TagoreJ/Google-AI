import streamlit as st
import PyPDF2
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import google.generativeai as genai
from dotenv import load_dotenv
# Used to securely store your API key
from google.colab import userdata # This is for Colab environment

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your key
# It's recommended to store your API key securely,
# for example, as a Streamlit Cloud secret.
# Replace 'GOOGLE_API_KEY' with the name of your secret.
# Use userdata.get('GOOGLE_API_KEY') for Colab, or os.getenv("GOOGLE_API_KEY") for local
try:
    api_key = userdata.get('GOOGLE_API_KEY')
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")


if not api_key:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY environment variable or Streamlit secret.")
else:
    genai.configure(api_key=api_key)

    st.title("PDF Question Answering with RAG and Gemini")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Initialize the Gemini model (moved outside the ask_question_rag function)
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        gemini_model = None

    # Define the RAG question answering function
    def ask_question_rag(user_question, gemini_model, vectorstore):
        """
        Answers a user question using RAG with Chroma and Gemini.

        Args:
            user_question (str): The question asked by the user.
            gemini_model: The initialized Gemini generative model.
            vectorstore: The initialized Chroma vector store.

        Returns:
            str: The generated answer from the Gemini model.
        """
        # Perform similarity search
        relevant_docs = vectorstore.similarity_search(user_question, k=3)

        # Combine relevant document content
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Construct the prompt for the Gemini model
        prompt = f"""
        You are a helpful assistant. Answer the following question based ONLY on the provided context.
        If you cannot find the answer in the context, please state that you cannot answer based on the provided information.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        # Generate the answer using the Gemini model
        response = gemini_model.generate_content(prompt)

        return response.text

    if uploaded_file is not None:
        st.write("Processing PDF...")
        progress_bar = st.progress(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        all_text = []

        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                all_text.append(text)
            progress = (i + 1) / total_pages
            progress_bar.progress(progress)
            # Add a small delay to make the progress bar visible for small PDFs
            # time.sleep(0.01) # Removed sleep for faster execution unless explicitly needed for demo

        processed_text = "\n".join(all_text)
        progress_bar.progress(1.0) # Ensure progress bar reaches 100% after PDF reading


        if processed_text:
            st.write("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            text_chunks = text_splitter.split_text(processed_text)
            st.write(f"Generated {len(text_chunks)} text chunks.")

            if text_chunks:
                st.write("Generating embeddings...")
                # Instantiate the embedding model
                embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                # Embeddings generation can be time-consuming, but direct progress update within
                # embed_documents is not trivial. We'll rely on the "Generating embeddings..." message
                # and the subsequent step's message to show progress.
                embeddings = embedding_model.embed_documents(text_chunks)
                st.write(f"Generated embeddings for {len(embeddings)} text chunks.")

                if embeddings:
                    st.write("Populating vector database...")
                    # Convert text chunks to Document objects
                    document_chunks = [Document(page_content=chunk) for chunk in text_chunks]

                    # Ensure directory for persistence exists
                    if not os.path.exists("./chroma_db"):
                        os.makedirs("./chroma_db")

                    # Initialize and populate the Chroma vector store
                    vectorstore = Chroma.from_documents(
                        documents=document_chunks,
                        embedding=embedding_model,
                        persist_directory="./chroma_db"
                    )

                    # Persist the database to disk
                    vectorstore.persist()

                    st.success("PDF processed, embedded, and vector database populated successfully!")
                    # A final progress bar update could be added here if a single bar was used for all steps,
                    # but with distinct messages, the success message serves as the final indicator.

                    # Add the Q&A interface after processing is complete
                    st.subheader("Ask a question about the PDF:")
                    user_question = st.text_input("Enter your question:")
                    ask_button = st.button("Get Answer")

                    if ask_button and user_question:
                        st.write("Searching for answer...")
                        try:
                            answer = ask_question_rag(user_question, gemini_model, vectorstore)
                            st.subheader("Answer:")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"An error occurred while getting the answer: {e}")
    elif 'uploaded_file' in locals() and uploaded_file is not None:
         st.warning("Please wait for the PDF processing and RAG setup to complete before asking questions.")
    else:
         st.info("Please upload a PDF to start.")