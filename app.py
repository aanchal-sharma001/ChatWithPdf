import uuid 
import streamlit as st 
from pathlib import Path 
import warnings  
import json 
from dotenv import load_dotenv 
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document 
from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings) 
from langchain_chroma import Chroma  
from pypdf import PdfReader  
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = Path("uploaded_pdfs")
DB_FOLDER = Path("chroma_db")
INDEX_FILE = Path("indexed_files.json")

PDF_FOLDER.mkdir(exist_ok=True) 
DB_FOLDER.mkdir(parents=True,exist_ok=True)
warnings.filterwarnings("ignore")




embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview") 

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,) # this is a chat model

#this function is used to  Save uploaded PDF files into the local PDF folder.
def save_uploaded_pdfs(uploaded_files):  
    for file in uploaded_files:
        file_path = PDF_FOLDER / file.name
        file_data = file.getvalue() #it returns the binary content
        if not file_path.exists() or file_path.read_bytes() != file_data:
            with open(file_path, "wb") as f:
                f.write(file_data)



def load_pdf(pdf_path):  
    reader = PdfReader(str(pdf_path))
    pages = [] 
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number 
                }))
    return pages 




def load_multiple_pdfs(pdf_paths):
    all_pages = []
    for pdf_path in pdf_paths: 
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)
    return all_pages 
  




def split_pages(pages): 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    return chunks




def create_vector_store(chunks):
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=str(DB_FOLDER)   
    )
    vector_store.add_documents(chunks)
    return vector_store

def load_vector_store(): 
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=str(DB_FOLDER)
    )
    return vector_store #existng collection access
  


def retrieve_chunks(question, selected_file): 
    search_kwargs = {"k": 4}
    if selected_file != "All documents":
        search_kwargs["filter"] = {
            "source": selected_file
        }
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs=search_kwargs
    )
    return retriever.invoke(question)




SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about uploaded PDF documents.

Rules:
1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. Add source references like [1], [2] after facts.
4. Mention the correct file name and page number.
5. If the answer is not available in the context, say:
   "I could not find this in the documents."
"""



## Formats retrieved document chunks with source, page number, and content
def format_context(docs):  #docs = retrieve_chunks(question, selected_file)
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        parts.append(
            f"[{i}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content: {doc.page_content}"
        )
    return "\n\n".join(parts) #retrieve chunks converted into text format



#purpose of this function is creating a list of source information from retrived chunks then we can show on ui 
def build_sources(docs):
    sources = []
    for i, doc in enumerate(docs, start=1):
        sources.append({
            "id": i, #chunk reference number
            "file": doc.metadata.get("source", "Unknown"), #pdf file name
            "page": doc.metadata.get("page", 0) + 1, #pdf page number
            "text": doc.page_content #retrieved chunk content
        })

    return sources



#retrieved pdf sources displayed on streamlit ui
def show_sources(sources):
    with st.expander("📚  View Sources & References", expanded=False):
        for source in sources:
            st.markdown(
                f"**[{source['id']}] {source['file']} · Page {source['page']}**"
            )
            st.caption(source["text"])
            if source != sources[-1]:
                st.divider()



def ask_model(question, docs, history):
    context = format_context(docs)
    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT + "\n\nContext:\n" + context
        )
    ]
    for message in history:
        if message["role"] == "user":
            messages.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":
            messages.append(
                AIMessage(content=message["content"])
            )

    messages.append(
        HumanMessage(content=question)
    )
    response = model.invoke(messages)
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        return "\n".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict) and "text" in block
        )
    return str(response.content)



st.set_page_config(
    page_title="Smart PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --purple: #6D4AFF;
    --purple-dark: #4B35C8;
    --ink: #17183B;
    --muted: #777995;
    --line: #E8E6FA;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #F8F7FF 0%, #F4F3FF 48%, #FFFFFF 100%);
    color: var(--ink);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #171A4A 0%, #20245F 58%, #392A83 100%);
    border-right: 1px solid rgba(255,255,255,.12);
}
[data-testid="stSidebar"] * {
    color: #F8F7FF !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: rgba(255,255,255,.09);
    border: 1px dashed rgba(255,255,255,.42);
    border-radius: 18px;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #7657FF;
    color: white !important;
    border: 0;
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 12px;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.16);
    border-radius: 14px;
}
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: rgba(255,255,255,.12);
    color: white !important;
    border: 1px solid rgba(255,255,255,.24);
    border-radius: 12px;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: rgba(255,255,255,.2);
    border-color: rgba(255,255,255,.5);
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.1rem 2.3rem;
    margin: 0 0 1.35rem 0;
    border-radius: 26px;
    background: linear-gradient(115deg, #E9E4FF 0%, #DCD5FF 52%, #BEB1FF 100%);
    border: 1px solid rgba(109,74,255,.15);
    box-shadow: 0 12px 35px rgba(91,70,190,.10);
}
.hero:after {
    content: "✦";
    position: absolute;
    right: 7%;
    top: 5%;
    font-size: 5rem;
    color: rgba(255,255,255,.65);
}
.hero h1 {
    margin: 0;
    color: #171744;
    font-size: clamp(1.8rem, 3vw, 2.7rem);
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -1.2px;
}
.hero h1 span {
    color: #6544E8;
}
.hero p {
    margin: .65rem 0 0 0;
    color: #41416F;
    font-size: 1.03rem;
}
.section-label {
    color: #777995;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 1.2rem 0 .65rem 0;
}
div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.92);
    border: 1px solid #E9E6FF;
    border-radius: 20px;
    padding: 1rem 1.15rem;
    box-shadow: 0 7px 22px rgba(47,42,105,.055);
    margin-bottom: .9rem;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #EEEAFE;
    border-color: #DCD4FF;
}
div[data-testid="stExpander"] {
    background: #F0EDFF;
    border: 1px solid #E1DBFF;
    border-radius: 15px;
}
div[data-testid="stExpander"] summary {
    color: #4933B5;
    font-weight: 700;
}
.stButton button {
    border-radius: 12px;
    font-weight: 700;
}
.stButton button[kind="primary"] {
    background: linear-gradient(100deg, #7655FF, #5639D5);
    border: 0;
    color: white;
}
[data-testid="stChatInput"] {
    border: 1px solid #D9D2FF;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(84,64,180,.08);
    background: white;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif;
}
hr {
    border-color: #E9E6F7;
}
@media (max-width: 700px) {
    .block-container { padding-top: 1rem; }
    .hero { padding: 1.4rem; border-radius: 20px; }
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div class="hero">
    <h1>Your <span>Smart PDF</span> Assistant</h1>
    <p>Get clear answers, explore your documents, and verify every response with source references.</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:12px 0 22px 0;'>"
        "<div style='font-size:3rem;'>📚</div>"
        "<div style='font-size:1.55rem;font-weight:800;letter-spacing:-.5px;'>PDF Chatbot</div>"
        "<div style='font-size:.9rem;opacity:.8;margin-top:5px;'>Your documents, one conversation away</div>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### ☁️ Upload your PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Save newly uploaded PDFs
    if uploaded_files:
        save_uploaded_pdfs(uploaded_files)

    # Get all saved PDF files
    pdf_paths = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_paths:
        st.info("Upload one or more PDF files to start chatting.")
        st.stop()

    st.markdown("---")
    st.markdown("### 🗂️ Your documents")

    file_names = [pdf.name for pdf in pdf_paths]
    selected_file = st.selectbox(
        "Choose a document",
        ["All documents"] + file_names
    )

    st.markdown("---")
    st.markdown("### 📊 Knowledge base")


# Load existing vector database
vector_store = load_vector_store()

# Load indexed files record
if INDEX_FILE.exists():
    with open(INDEX_FILE, "r") as f:
        try:
            indexed_files = json.load(f)
        except json.JSONDecodeError:
            indexed_files = {}
else:
    indexed_files = {}

# Check which PDFs are new or modified
with st.spinner("Preparing your documents..."):
    for pdf_path in pdf_paths:
        file_info = {
            "size": pdf_path.stat().st_size,
            "modified": pdf_path.stat().st_mtime_ns
        }
        saved_info = indexed_files.get(pdf_path.name, {})
        if saved_info.get("size") != file_info["size"] or \
           saved_info.get("modified") != file_info["modified"]:
            # Remove old chunks if this PDF was indexed earlier
            vector_store.delete(
                where={"source": pdf_path.name}
            )
            # Read PDF
            pages = load_pdf(pdf_path)
            # Split into chunks
            chunks = split_pages(pages)
            # Add new chunks to database
            if chunks:
                vector_store.add_documents(chunks)
            # Update record
            indexed_files[pdf_path.name] = {
                "size": file_info["size"],
                "modified": file_info["modified"],
                "pages": len(pages),
                "chunks": len(chunks)
            }

    # Save indexing record
    with open(INDEX_FILE, "w") as f:
        json.dump(indexed_files, f, indent=4)

# Store vector database in session
st.session_state.vector_store = vector_store

# Calculate total pages from PDFs
st.session_state.num_pages = sum(
    indexed_files[p.name]["pages"] for p in pdf_paths
)

# Calculate total chunks from PDFs
st.session_state.num_chunks = sum(
    indexed_files[p.name]["chunks"] for p in pdf_paths
)



with st.sidebar:
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);
                    border-radius:15px;padding:14px;margin-top:8px;">
            <div style="font-size:.82rem;opacity:.8;">DOCUMENTS READY</div>
            <div style="font-size:1.5rem;font-weight:800;margin:5px 0;">
                {len(pdf_paths)} <span style="font-size:.9rem;font-weight:500;">PDF files</span>
            </div>
            <div style="font-size:.86rem;opacity:.9;">
                📄 {st.session_state.num_pages} pages &nbsp;·&nbsp;
                🧩 {st.session_state.num_chunks} chunks
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()



if not st.session_state.messages:
    st.markdown(
        "<div style='text-align:center;padding:1.3rem 1rem 1rem 1rem;'>"
        "<div style='font-size:2.4rem;'>🤖</div>"
        "<div style='font-size:1.35rem;font-weight:800;color:#22204F;'>What would you like to know?</div>"
        "<div style='color:#8585A1;margin-top:6px;'>Ask a question about the information inside your PDFs.</div>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-label">YOUR CONVERSATION</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="section-label">YOUR CONVERSATION</div>', unsafe_allow_html=True)



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])




question = st.chat_input("Ask anything about your PDFs...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    docs = retrieve_chunks(question, selected_file)

    if not docs:
        answer = "I could not find this in the selected documents."
        sources = []
    else:
        with st.spinner("Reading your documents and preparing an answer..."):
            answer = ask_model(
                question,
                docs,
                st.session_state.messages
            )
            sources = build_sources(docs)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
