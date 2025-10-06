import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your key
# It's recommended to store your API key securely,
# for example, as a Streamlit Cloud secret.
# Replace 'GOOGLE_API_KEY' with the name of your secret.
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY environment variable or Streamlit secret.")
else:
    genai.configure(api_key=api_key)

    st.title("Gemini API Question Answering App")

    # Initialize the Gemini model
    # You can change the model name if needed
    try:
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        gemini_model = None


    if gemini_model:
        # Get user input
        user_question = st.text_input("Ask a question:")

        if user_question:
            st.write("Getting answer from Gemini...")

            try:
                # Generate a response
                response = gemini_model.generate_content(user_question)

                # Display the answer
                st.write("Answer:")
                st.write(response.text)

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")