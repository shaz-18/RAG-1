# Add this at the top of your file
import base64
import streamlit as st

st.set_page_config(
    page_title="AutoBot",
    page_icon="📚",
    layout="wide"
)
# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007BFF;  /* Changed to a bright blue */
        color: white;
        border: none;
        margin: 10px 0;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border-color: #007BFF;  /* Changed border color to blue */
    }
    .upload-text {
        font-size: 1.2em;
        color: #0056b3;  /* Changed to a darker blue */
        margin-bottom: 1em;
    }
    .success-message {
        padding: 1em;
        border-radius: 10px;
        background-color: #E0F7FA;  /* Softer light blue background */
        border-left: 5px solid #007BFF;  /* Blue accent */
    }
    .error-message {
        padding: 1em;
        border-radius: 10px;
        background-color: #FFEBEE;  /* Softer light red background */
        border-left: 5px solid #C62828;  /* Richer red accent */
    }
</style>
""", unsafe_allow_html=True)

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import docx2txt

# Load environment variables and configure API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Helper Functions

@st.cache_data
def get_document_text(docs):
    """Extract text from various document types including PDFs and Word documents."""
    text = ""
    try:
        for doc in docs:
            file_extension = doc.name.split('.')[-1].lower()
            
            if file_extension in ['pdf']:
                # Handle PDFs
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            
            elif file_extension in ['docx']:
                # Handle Word documents
                doc_text = docx2txt.process(doc)
                text += doc_text
            
    except Exception as e:
        st.error(f"Error processing document {doc.name}: {e}")
    return text

@st.cache_data
def get_text_chunks(text):
    """Split the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

@st.cache_data
def get_vector_store(text_chunks):
    """Convert text chunks into vector embeddings and store them in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Simplified prompt template
def get_conversational_chain():
    """Set up the LLM-based QA chain using Google's Gemini model."""
    prompt_template = """
    You are a helpful academic advisor. Answer the following question based on the context provided.
    If the answer isn't in the context, say "I don't have enough information to answer that question."
    
    Context: {context}
    Question: {question}
    Answer:"""
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def user_input(user_question):
    """Process user questions and return answers using FAISS index."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=5)

        if not docs:
            return "No relevant information found in the provided context."

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        st.error(f"Error fetching answer: {e}")
        return None

# Display PDF in the Streamlit app
def displayPDF(file):
    """Display a PDF preview or provide a download link for large files."""
    try:
        # Encode the PDF to base64
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')

        # Embed the PDF in an iframe for preview (if possible)
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="600" type="application/pdf">
        </iframe>
        """

        # Display the PDF preview
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception:
        # If preview fails, provide a download link as fallback
        st.warning("Preview not available. Please download the file to view.")
        file.seek(0)  # Reset file pointer for download
        st.download_button(label="Download PDF", data=file, 
                           file_name=file.name, mime="application/pdf")

# Enhanced chatbot page
def chatbot_page():
    """Page for uploading PDFs and asking questions."""
    st.header("🎓 AutoBot")
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style = "color: white"; class="upload-text">
            📚 Upload your academic documents and ask questions!
        </div>
        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'docx'],  # Only PDF and DOCX file types
            help="You can upload PDFs and Word documents"
        )

    with col2:
        if pdf_docs:
            st.info(f"📑 {len(pdf_docs)} files uploaded")
            
    if pdf_docs:
        with st.expander("View Uploaded Documents"):
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")
                displayPDF(pdf)

    # Process button with loading animation
    if st.button("🚀 Process Documents", key="process_btn"):
        if not pdf_docs:
            st.error("Please upload PDF files first!")
        else:
            with st.spinner("🔄 Processing your documents..."):
                progress_bar = st.progress(0)
                
                # Process documents with progress updates
                raw_text = get_document_text(pdf_docs)
                progress_bar.progress(33)
                
                text_chunks = get_text_chunks(raw_text)
                progress_bar.progress(66)
                
                get_vector_store(text_chunks)
                progress_bar.progress(100)
                
                st.session_state['pdf_processed'] = True
                st.markdown("""
                <div style="color: black;" class="success-message">
                    ✅ Documents processed successfully!
                </div>
                """, unsafe_allow_html=True)

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for role, message in st.session_state.chat_history:
            if role == "You":
                message_container = st.container()
                with message_container:
                    st.markdown(f"""
                        <div style="background-color: black; padding: 15px; 
                                border-radius: 15px; margin: 5px 0; max-width: 80%; 
                                float: right; clear: both;">
                            <strong>You:</strong><br>{message}
                        </div>""", 
                        unsafe_allow_html=True
                    )
            else:
                message_container = st.container()
                with message_container:
                    st.markdown(f"""
                        <div style="background-color: black; padding: 15px; 
                                border-radius: 15px; margin: 5px 0; max-width: 80%; 
                                float: left; clear: both;">
                            <strong>Assistant:</strong><br>{message}
                        </div>""", 
                        unsafe_allow_html=True
                    )

    # Question input
    if st.session_state.get('pdf_processed', False):
        st.markdown("### Ask Your Question")
        
        # Create a form for input
        with st.form(key='question_form', clear_on_submit=True):
            user_question = st.text_input(
                "",
                placeholder="Type your question here...",
                key="question_input"
            )
            submit_button = st.form_submit_button("Send")

            if submit_button and user_question:
                with st.spinner("🤔 Thinking..."):
                    answer = user_input(user_question)
                    if answer:
                        # Update chat history
                        st.session_state.chat_history.append(("You", user_question))
                        st.session_state.chat_history.append(("Assistant", answer))
                        st.rerun()

        # Add clear history button outside the form
        if st.session_state.chat_history:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

# Enhanced home page
def home_page():
    st.title("🏛️ Automated Information Retrieval System")
    
    # Create three columns for features with borders and padding
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="border: 2px solid #007BFF; border-radius: 10px; padding: 15px;">
            <h3 style="color: #007BFF;">📚 Easy Upload</h3>
            <p>Upload multiple PDFs and get instant access to their contents.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border: 2px solid #007BFF; border-radius: 10px; padding: 15px;">
            <h3 style="color: #007BFF;">🔍 Fast Search</h3>
            <p>Find information across all your documents instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="border: 2px solid #007BFF; border-radius: 10px; padding: 15px;">
            <h3 style="color: #007BFF;">🤖 AI-Powered Bot</h3>
            <p>Get answers to your questions using advanced AI technology.</p>
        </div>
        """, unsafe_allow_html=True)

    st.image("wceimage.jpg", use_container_width=True)

    # Additional project description
    st.markdown("""
    <div style="border: 2px solid #007BFF; border-radius: 10px; padding: 15px; margin-top: 20px;">
        <h3 style="color: #007BFF;">About This Project</h3>
        <p>This project leverages cutting-edge AI technology to help students and researchers efficiently retrieve academic information from their documents. 
        With a user-friendly interface, you can easily upload your PDFs and get instant answers to your questions, making your research process smoother and more effective.</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced about page
def about_page():
    st.header(" About This Project")

    st.write("""
    This project helps students retrieve academic information from PDFs using 
    Google's Gemini model for efficient responses. Our goal is to streamline the research process, making it easier for users to access and understand complex information.
    """)

    st.markdown("""
    <div style="background-color: #EBF5FB; padding: 20px; 
                border-radius: 10px; border-left: 5px solid #3498DB;">
        <h3 style="color: #2C3E50;">Features</h3>
        <ul style="color: #34495E;">
            <li>PDF Processing</li>
            <li>Advanced AI-powered Question Answering</li>
            <li>Multi-document Search</li>
            <li>User-friendly Interface</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 20px;">
        <h3 style="color: #007BFF;">Technologies Used</h3>
        <p>This project leverages several cutting-edge technologies:</p>
        <ul>
            <li><strong>Streamlit:</strong> A powerful framework for building interactive web applications in Python.</li>
            <li><strong>LangChain:</strong> A framework for developing applications powered by language models.</li>
            <li><strong>Google Generative AI:</strong> Utilizes Google's advanced AI models for natural language processing and understanding.</li>
            <li><strong>PyPDF2:</strong> A library for reading and manipulating PDF files.</li>
            <li><strong>Docx2txt:</strong> A tool for extracting text from Word documents.</li>
        </ul>
    </div>
    """,unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 20px;">
        <h3 style="color: #007BFF;">Future Enhancements</h3>
        <p>We plan to implement additional features such as:</p>
        <ul>
            <li>Support for more document formats (e.g., TXT, HTML).</li>
            <li>Enhanced user interface for better user experience.</li>
            <li>Integration with external databases for broader information retrieval.</li>
        </ul>
    </div>
    """,unsafe_allow_html=True)

def main():
    """Main function for navigation."""
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home'

    # Add logo to the sidebar
    st.sidebar.image("download.jpeg", use_container_width=True)  # Adjust the path as necessary
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("Home"):
        st.session_state['page'] = 'Home'
    if st.sidebar.button("Chatbot"):
        st.session_state['page'] = 'Chatbot'
    if st.sidebar.button("About"):
        st.session_state['page'] = 'About'

    if st.session_state['page'] == 'Home':
        home_page()
    elif st.session_state['page'] == 'Chatbot':
        chatbot_page()
    elif st.session_state['page'] == 'About':
        about_page()

if __name__ == "__main__":
    main()
