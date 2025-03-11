import streamlit as st
import os

# pip install streamlit langchain huggingface_hub sentence-transformers faiss-cpu

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI API & HALAMAN
# ─────────────────────────────────────────────────────────────────────────────

# Ganti GROQ_API_KEY dengan kunci Anda sendiri, misalnya di secrets.toml
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="Psy-Mon",
    page_icon="📓",
    layout="wide"
)

# CSS Styling
st.markdown(
    """
    <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #f0f2f6; }
        .bot-message { background-color: #e8f0fe; }
        .reset-button {
            background-color: #ff4b4b;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.3rem;
            border: none;
            cursor: pointer;
            margin-top: 1rem;
        }
        .reset-button:hover {
            background-color: #ff0000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("Psychology of Money")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan Psikologi Uang.
    ChatBot ini akan membantu Anda memahami lebih dalam Psikologi Uang dari sudut pandang Addison Bell, Morgan Housel, dan Jim Ware. **Melalui apps ini Anda seperti memiliki 3 mentor kawakan** yang senantiasa siap menjawab pertanyaan berbagai pertanyaan Anda  tentang psikologi uang dan investasi. Cara terbaik memanfaatkan ChatBot ini adalah dengan membeli dan membaca 3 bukunya, agar Anda dapat mengajukan pertanyaan spesifik tingkat lanjutan terkait subject matternya. **SEMOGA BERMANFAAT!**
    """
)

# Membuat kolom untuk menampung gambar secara horizontal
col1, col2, col3 = st.columns(3)

# Menampilkan gambar di setiap kolom dengan ukuran yang telah disesuaikan
with col1:
    st.image("images/psymon_addisonBell.jpg", caption="Addison Bell", width=126)

with col2:
    st.image("images/psymon_morganHousel.jpg", caption="Morgan Housel", width=126)

with col3:
    st.image("images/psymon_jimware.jpg", caption="Jim Ware", width=126)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE DAN INISIALISASI
# ─────────────────────────────────────────────────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:
    st.session_state.chain = None

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA
# ─────────────────────────────────────────────────────────────────────────────
# Prompt ini akan memaksa jawaban selalu dalam Bahasa Indonesia.
PROMPT_INDONESIA = """\
Gunakan informasi konteks berikut untuk menjawab pertanyaan pengguna selalu dalam bahasa Indonesia yang baik dan terstruktur.
Selalu berikan jawaban terbaik yang dapat kamu berikan dalam bahasa indonesia.

Konteks: {context}
Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_INDONESIA
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNGSI INISIALISASI RAG
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_rag():
    """
    Memuat dokumen PDF dari folder 'documents', memecah menjadi chunk,
    membuat FAISS vector store, dan membentuk ConversationalRetrievalChain.
    """
    try:
        # 4.1 Load Dokumen PDF
        loader = DirectoryLoader("documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # 4.2 Split Dokumen
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1008, chunk_overlap=234)
        texts = text_splitter.split_documents(documents)

        # 4.3 Embedding Berbahasa Indonesia
        # Ganti sesuai preferensi, misal "indobenchmark/indobert-base-p1", dsb.
        embeddings = HuggingFaceEmbeddings(
            model_name="LazarusNLP/all-indo-e5-small-v4",
            model_kwargs={'device': 'cpu'}
        )

        # 4.4 Membuat Vector Store FAISS
        vectorstore = FAISS.from_documents(texts, embeddings)

        # 4.5 Menginisialisasi LLM (ChatGroq)
        llm = ChatGroq(
            temperature=0.54,
            model_name="gemma2-9b-it",
            max_tokens=1024
        )

        # 4.6 Membuat Memory untuk menyimpan riwayat percakapan
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        # 4.7 Membuat ConversationalRetrievalChain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                'prompt': INDO_PROMPT_TEMPLATE,
                'output_key': 'answer'
            }
        )

        return chain

    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNGSI RESET CHAT DAN MEMORY
# ─────────────────────────────────────────────────────────────────────────────
def reset_chat_and_memory():
    """Reset chat history and conversation memory"""
    st.session_state.chat_history = []
    if st.session_state.chain and hasattr(st.session_state.chain, 'memory'):
        st.session_state.chain.memory.clear()
    st.success("Chat dan memory berhasil direset!")
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 6. INISIALISASI SISTEM
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain is None:
    with st.spinner("Memuat sistem..."):
        st.session_state.chain = initialize_rag()

# ─────────────────────────────────────────────────────────────────────────────
# 7. ANTARMUKA CHAT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain:
    # 7.1 Tampilkan riwayat chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 7.2 Chat Input
    prompt = st.chat_input("✍️tuliskan pertanyaan Anda tentang psikologi uang dan investasi di sini")
    if prompt:
        # Tambahkan pertanyaan user ke riwayat chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 7.3 Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    # Panggil chain
                    result = st.session_state.chain.invoke({"question": prompt})
                    # Ambil jawaban
                    answer = result.get('answer', '')
                    st.write(answer)
                    # Tambahkan ke riwayat
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# ─────────────────────────────────────────────────────────────────────────────
# 8. FOOTER & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **Disclaimer:**
    - Sistem ini menggunakan AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.
    - Ketik: LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. TOMBOL RESET CHAT DAN MEMORY
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Reset Chat dan Memory", key="reset_button", type="primary", use_container_width=True):
        reset_chat_and_memory()
