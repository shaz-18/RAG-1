# 📄 PDF Chatbot using LangChain, FAISS, and Google Generative AI

This project is a document-based chatbot built using Streamlit, LangChain, FAISS, and Google Generative AI. It allows users to upload PDF and DOCX files, processes the content, and enables question-answering from the documents through an interactive web interface.

## 🚀 Features
- **Document Upload:** Supports both PDF and DOCX formats.
- **Intelligent Querying:** Uses Google Generative AI for accurate responses.
- **Efficient Data Handling:** FAISS for fast and scalable vector search.
- **Web Interface:** Built with Streamlit for a simple, user-friendly experience.

## 📦 Tech Stack
- **Python** (Core language)
- **Streamlit** (Web Interface)
- **PyPDF2 & docx2txt** (Document Parsing)
- **LangChain** (Text Splitting and Query Handling)
- **FAISS** (Vector Storage and Search)
- **Google Generative AI & LangChain Integration**
- **dotenv** (Environment Variable Management)

## 📁 Project Structure
```plaintext
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── README.md               # Project documentation
├── data/                   # Sample documents for testing
└── utils/                  # Helper functions (if any)
```

## 📦 Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   .\venv\Scripts\activate   # For Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the `.env` file:**
   - Create a `.env` file in the root directory with the following content:
     ```plaintext
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## ▶️ Usage
1. **Run the application:**
   ```bash
   streamlit run app.py
   ```
2. **Upload a PDF or DOCX file** and enter your query in the provided text box.
3. **Get instant answers** powered by Google Generative AI and FAISS.

## 📊 How It Works
1. **Document Upload:** Users upload their documents via the Streamlit UI.
2. **Text Extraction:** `PyPDF2` and `docx2txt` extract the content from the documents.
3. **Text Splitting:** The content is split using `LangChain` for better chunking and embeddings.
4. **Vector Storage:** FAISS stores the document chunks as vectors for efficient retrieval.
5. **Query Processing:** User questions are embedded and compared against the stored vectors.
6. **Answer Generation:** The closest vector matches are sent to the Google Generative AI for response generation.

