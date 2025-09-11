🧠 PDF Chatbot (RAG-based)

A Retrieval-Augmented Generation (RAG) chatbot that allows you to upload PDF, DOCX, TXT, or MD files and interactively ask questions. Built with FastAPI for the backend, Streamlit for the frontend, and Cerebras LLMs for AI responses.

Features:
- Upload multiple PDFs, DOCX, or text files.
- Automatic text extraction and chunking.
- Semantic search using SentenceTransformers embeddings.
- Memory of recent conversation turns (SQLite).
- LLM-powered responses with context-aware answers.
- Shows citations for retrieved document chunks.
- Reset conversation memory at any time.

Tech Stack:
- Backend: FastAPI, Python, Cerebras API, SentenceTransformers
- Frontend: Streamlit
- Database: SQLite (for conversation memory)
- File Processing: PyPDF, Docx2txt
- Deployment: Can be hosted on free platforms like Render, Railway, or Streamlit Cloud

Setup Instructions:

1. Clone the repository:
   git clone <your-repo-url>
   cd <your-repo>

2. Backend Setup:
   cd backend
   python -m venv venv
   # Activate virtual environment
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   pip install -r requirements.txt

   Configure Environment Variables:
   Create a .env file in the backend folder:

   CS_API_KEY=your_cerebras_api_key
   LLM_MODEL=your_cerebras_model_name
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   MEMORY_TURNS=8
   BACKEND_URL=http://localhost:8000

   Start the backend server:
   uvicorn main:app --reload

3. Frontend Setup:
   cd ../frontend
   python -m venv venv
   # Activate virtual environment
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   pip install -r requirements.txt

   Create a .env file in the frontend folder (optional):
   BACKEND_URL=http://localhost:8000

   Run Streamlit:
   streamlit run app.py

Usage:
1. Open the frontend URL in your browser (usually http://localhost:8501).
2. Upload your PDF, DOCX, or TXT files.
3. Ask questions in the chat box — answers will include citations from uploaded files.
4. Adjust Top K chunks in the sidebar to control retrieval size.
5. Reset memory anytime from the sidebar.

Notes:
- The chatbot uses the Cerebras LLM API, which requires an API key.
- Free Cerebras API keys have usage limits; check your Cerebras account for details.
- .env files and data/ folders are ignored in Git to protect sensitive info.

Folder Structure:
project-root/
├── backend/
│   ├── main.py
│   ├── data/          # embeddings, memory DB, uploaded files (ignored in Git)
│   └── .env           # Cerebras API key & config
├── frontend/
│   ├── app.py
│   └── .env           # optional backend URL
├── .gitignore
└── README.txt

License:
MIT License – free to use, modify, and distribute.
