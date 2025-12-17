import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# --- Download NLTK resources (hanya berjalan sekali) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ==========================================
# 1. FUNGSI HELPER (Proses Data)
# ==========================================

def extract_text_from_pdf(uploaded_file):
    """Mengekstrak teks mentah dari file PDF yang diupload."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text

def clean_and_tokenize(text):
    """Membersihkan teks, menghapus stopwords, dan tokenisasi."""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Hapus karakter non-huruf (angka, simbol)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenisasi
    tokens = word_tokenize(text)
    
    # 4. Filter Stopwords (Ubah 'english' ke 'indonesian' jika perlu)
    stop_words = set(stopwords.words('english')) 
    
    # Tambahkan custom stopword jika perlu (misal: 'et', 'al' untuk paper)
    custom_stops = {'et', 'al', 'fig', 'table', 'data', 'using', 'based', 'model'} 
    stop_words.update(custom_stops)
    
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return filtered_tokens

def build_cooccurrence_graph(tokens, window_size=3):
    """Membangun graph co-occurrence dari list token."""
    graph = nx.Graph()
    
    # Loop melalui token dengan sliding window
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        
        # Hubungkan setiap kata dalam window satu sama lain
        for j in range(len(window)):
            for k in range(j + 1, len(window)):
                w1, w2 = sorted([window[j], window[k]])
                if w1 != w2:
                    if graph.has_edge(w1, w2):
                        graph[w1][w2]['weight'] += 1
                    else:
                        graph.add_edge(w1, w2, weight=1)
    return graph

# ==========================================
# 2. VISUALISASI (Menggunakan Kode Sebelumnya)
# ==========================================

def plot_graph(graph, pagerank_scores, threshold=2):
    """Fungsi plotting yang sudah dimodifikasi dengan ketebalan dinamis."""
    
    # Buat figure baru (Ukuran disesuaikan agar muat di web)
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Layout
    pos = nx.spring_layout(graph, k=0.15, iterations=50, seed=42)
    
    # Node Size berdasarkan PageRank
    # Skala dikurangi sedikit agar tidak terlalu menutupi layar web
    node_sizes = [v * 30000 for v in pagerank_scores.values()]
    
    # Draw Nodes
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, ax=ax)
    
    # --- EDGE DYNAMIC WIDTH ---
    edges = graph.edges(data=True)
    weights = [data['weight'] for u, v, data in edges]
    max_weight = max(weights) if weights else 1
    
    # Normalisasi ketebalan
    edge_widths = [(w / max_weight) * 4 for w in weights]
    
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.4, edge_color='gray', ax=ax)
    
    # --- LABELS ---
    # Label Node
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Label Edge (Hanya tampilkan jika weight >= threshold)
    edge_labels = {}
    for u, v, data in edges:
        if data['weight'] >= threshold:
            edge_labels[(u, v)] = str(data['weight'])
            
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', font_size=8, ax=ax)
    
    ax.axis('off')
    return fig

# ==========================================
# 3. LOGIKA UTAMA STREAMLIT
# ==========================================

st.set_page_config(layout="wide", page_title="Paper Keyword Analyzer")

st.title("📄 Paper Keyword Extractor & Graph Visualization")
st.markdown("Upload file PDF paper, aplikasi akan mengekstrak kata kunci penting menggunakan **PageRank** dan memvisualisasikan hubungannya.")

# Sidebar untuk upload dan setting
with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")
    
    st.header("Settings")
    window_size = st.slider("Window Size (Co-occurrence)", 2, 5, 3)
    top_n_words = st.slider("Jumlah Kata Kunci Ditampilkan", 5, 20, 10)
    edge_threshold = st.slider("Threshold Label Edge (Minimal Bobot)", 1, 10, 2)

if uploaded_file is not None:
    # 1. Ekstraksi Teks
    with st.spinner('Mengekstrak teks dari PDF...'):
        raw_text = extract_text_from_pdf(uploaded_file)
        
    # Tampilkan preview teks (opsional)
    with st.expander("Lihat Teks Asli (Preview 500 karakter)"):
        st.text(raw_text[:500] + "...")
        
    # 2. Preprocessing
    tokens = clean_and_tokenize(raw_text)
    st.success(f"Berhasil memproses {len(tokens)} kata bersih.")

    # 3. Build Graph & PageRank
    if len(tokens) > 0:
        with st.spinner('Membangun Graph dan Menghitung PageRank...'):
            graph = build_cooccurrence_graph(tokens, window_size=window_size)
            pagerank_scores = nx.pagerank(graph, weight='weight')
            
            # Urutkan keyword
            sorted_keywords = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        # --- TAMPILKAN HASIL ---
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"Top {top_n_words} Keywords")
            # Buat DataFrame untuk tampilan rapi
            df_keywords = pd.DataFrame(sorted_keywords[:top_n_words], columns=["Kata Kunci", "Score PageRank"])
            st.dataframe(df_keywords, use_container_width=True)
            
        with col2:
            st.subheader("Visualisasi Graph")
            st.caption(f"Menampilkan hubungan antar kata. Ketebalan garis = kekuatan hubungan.")
            # Render plot
            fig = plot_graph(graph, pagerank_scores, threshold=edge_threshold)
            st.pyplot(fig)
            
    else:
        st.warning("Tidak ada teks yang valid ditemukan setelah pembersihan. Coba periksa file PDF.")