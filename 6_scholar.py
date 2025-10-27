import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import networkx as nx
import io
import re

# =====================
# APP CONFIG
# =====================
st.set_page_config(layout="wide", page_title="Topic Map dari Kolom title")
st.title("🗺️ Topic Map dari Kolom `title`")

st.markdown(
    """
Aplikasi ini membaca dataset dengan skema kolom seperti:
**author, title, num_citations, number_of_co_authors, pub_year, 2000, 2001, ..., 2022**.
Fokus utama: *topic modeling* dari kolom **title** dan visualisasi **topic map**.
    """
)

# =====================
# UTILITIES
# =====================
INDO_STOPWORDS = set([
    'yang','dan','di','ke','dari','pada','dengan','untuk','adalah','ini','itu','dalam','atau','oleh',
    'juga','sebagai','karena','tidak','dapat','akan','agar','lebih','bagi','terhadap','menggunakan',
    'pembuatan','meningkatkan','kegiatan','berupa','pemanfaatan','berbasis','pada','hasil','studi',
    'analisis','kajian','pengaruh','model','metode','pendekatan','sistem','data','pengembangan'
])

EN_STOPWORDS = set([
    'the','and','of','to','in','for','on','with','by','from','or','as','an','a','is','are','using','use',
    'based','analysis','study','effect','model','method','approach','system','data','development'
])

CUSTOM_STOPWORDS = INDO_STOPWORDS | EN_STOPWORDS

YEAR_COLS = [str(y) for y in range(2000, 2023)]  # 2000..2022

@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Format file tidak didukung. Gunakan .xlsx, .xls, atau .csv")
        return None
    return df


def sanitize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    # Lowercase + remove non-letter/number + collapse spaces
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\u00C0-\u024F\u1E00-\u1EFF\s]", " ", s)  # keep latin & accents
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_vectorizer(max_df=0.95, min_df=2, ngram_range=(1,2)):
    return TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=list(CUSTOM_STOPWORDS)
    )


def fit_nmf(tfidf_matrix, n_components=6, random_state=42):
    nmf_model = NMF(n_components=n_components, random_state=random_state, init='nndsvda')
    W = nmf_model.fit_transform(tfidf_matrix)  # doc-topic
    H = nmf_model.components_                 # topic-term
    return nmf_model, W, H


def top_terms_for_topics(H, feature_names, topk=12):
    topics = []
    for t_idx, t_vec in enumerate(H):
        top_idx = np.argsort(t_vec)[::-1][:topk]
        terms = [feature_names[i] for i in top_idx]
        scores = [t_vec[i] for i in top_idx]
        topics.append({"topic": t_idx, "terms": terms, "scores": scores})
    return topics


def make_topic_graph(topics_list):
    G = nx.Graph()
    for t in topics_list:
        t_name = f"Topic {t['topic']+1}"
        G.add_node(t_name, bipartite='topic')
        for term in t['terms']:
            if not G.has_node(term):
                G.add_node(term, bipartite='term')
            G.add_edge(t_name, term)
    return G


# =====================
# SIDEBAR
# =====================
st.sidebar.header("⚙️ Pengaturan")
st.sidebar.write("Upload file dan atur parameter analisis.")

uploaded_file = st.sidebar.file_uploader("Upload file (.xlsx/.csv)", type=["xlsx","xls","csv"])

n_components = st.sidebar.slider("Jumlah Topik (k)", 2, 15, 6, 1)
max_df = st.sidebar.slider("max_df (filter kata terlalu umum)", 0.60, 1.00, 0.95, 0.01)
min_df = st.sidebar.number_input("min_df (dokumen minimum)", min_value=1, value=2, step=1)
ngram = st.sidebar.select_slider("n-gram", options=["1","1-2","1-3"], value="1-2")
ngram_map = {"1":(1,1), "1-2":(1,2), "1-3":(1,3)}

show_wordcloud = st.sidebar.checkbox("Tampilkan Wordcloud", value=True)

# =====================
# DATA LOAD & BASIC CHECKS
# =====================
if uploaded_file:
    df = load_file(uploaded_file)
    if df is not None:
        # Standardize expected columns if present
        expected_cols = [
            'author','title','num_citations','number_of_co_authors','pub_year'
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.warning(f"Kolom berikut tidak ditemukan: {missing}. Pastikan nama kolom sesuai.")
        
        # Keep only needed cols for modeling
        if 'title' not in df.columns:
            st.error("Kolom 'title' wajib ada untuk topic modeling.")
            st.stop()

        work = df.copy()
        work['title_clean'] = work['title'].apply(sanitize_text)
        work = work[work['title_clean'].str.len() > 0].reset_index(drop=True)

        # Simple EDA header
        st.subheader("🔍 Ringkasan Data")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Jumlah Dokumen", len(work))
        with col_b:
            if 'author' in work.columns:
                st.metric("Jumlah Penulis Unik", work['author'].nunique())
            else:
                st.metric("Jumlah Penulis Unik", "-")
        with col_c:
            if 'pub_year' in work.columns:
                st.metric("Rentang Tahun", f"{int(work['pub_year'].min())}–{int(work['pub_year'].max())}")
            else:
                st.metric("Rentang Tahun", "-")

        # =====================
        # TOPIC MODELING
        # =====================
        st.subheader("🧠 Topic Modeling dari Kolom title")
        vectorizer = build_vectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_map[ngram])
        tfidf = vectorizer.fit_transform(work['title_clean'])

        nmf_model, W, H = fit_nmf(tfidf, n_components=n_components)
        feature_names = vectorizer.get_feature_names_out()
        topics = top_terms_for_topics(H, feature_names, topk=12)

        # tampilkan daftar topik + wordcloud opsional
        topic_rows = []
        cols = st.columns(2)
        for idx, t in enumerate(topics):
            sentence = f"Topik {t['topic']+1}: {', '.join(t['terms'][:10])}"
            topic_rows.append({"Topik": f"Topik {t['topic']+1}", "Kata Kunci": ', '.join(t['terms'][:10])})
            with cols[idx % 2]:
                st.markdown(f"**{sentence}**")
                if show_wordcloud:
                    wc = WordCloud(width=800, height=300, background_color='white').generate(' '.join(t['terms']))
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

        # =====================
        # TOPIC MAP (Topic-Term Graph)
        # =====================
        st.subheader("🕸️ Topic Map (Graf Topik–Kata)")
        G = make_topic_graph(topics)
        fig, ax = plt.subplots(figsize=(11, 8))
        pos = nx.spring_layout(G, k=0.7, seed=42)
        node_colors = ['#8ecae6' if G.nodes[n].get('bipartite') == 'term' else '#ffb703' for n in G.nodes]
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=900,
            node_color=node_colors,
            edge_color='#999999',
            font_size=9,
            ax=ax
        )
        st.pyplot(fig)

        # =====================
        # DOKUMEN x TOPIK (Heatmap)
        # =====================
        st.subheader("🌡️ Distribusi Dokumen × Topik (Top 100 dokumen)")
        top_docs = min(100, W.shape[0])
        plt.figure()
        plt.clf()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.heatmap(W[:top_docs, :], cmap="YlGnBu", cbar=True, ax=ax2)
        ax2.set_xlabel("Topik")
        ax2.set_ylabel("Dokumen (subset)")
        st.pyplot(fig2)

        # =====================
        # RINGKASAN PER AUTHOR (opsional)
        # =====================
        if 'author' in work.columns:
            st.subheader("👤 Ringkasan Topik per Penulis (Top 15 berdasarkan jumlah dokumen)")
            # Ambil top penulis
            top_authors = work['author'].value_counts().head(15).index
            author_df = work.loc[work['author'].isin(top_authors), ['author']].copy()
            for k in range(n_components):
                author_df[f'topic_{k+1}'] = 0.0

            # Agregasi skor topik per author
            for i, row in work.loc[work['author'].isin(top_authors)].reset_index(drop=True).iterrows():
                a = row['author']
                w_vec = W[i, :]
                author_df.loc[author_df['author'] == a, [f'topic_{k+1}' for k in range(n_components)]] += w_vec

            author_df = author_df.groupby('author', as_index=False).sum()
            # Normalisasi per author
            topic_cols = [c for c in author_df.columns if c.startswith('topic_')]
            author_df[topic_cols] = author_df[topic_cols].div(author_df[topic_cols].sum(axis=1), axis=0).fillna(0)

            # Plot heatmap
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(author_df.set_index('author')[topic_cols], cmap='YlOrBr', ax=ax3)
            ax3.set_xlabel('Topik')
            ax3.set_ylabel('Author')
            st.pyplot(fig3)

            # Unduh ringkasan author
            buf_auth = io.BytesIO()
            with pd.ExcelWriter(buf_auth, engine='openpyxl') as writer:
                author_df.to_excel(writer, index=False, sheet_name="Author_Topic")
            st.download_button("📥 Unduh Ringkasan Author × Topik", data=buf_auth.getvalue(), file_name="author_topic_summary.xlsx")

        # =====================
        # TREND TAHUNAN (pub_year & kolom 2000..2022)
        # =====================
        st.subheader("📈 Tren Tahunan")
        lay1, lay2 = st.columns(2)
        with lay1:
            if 'pub_year' in work.columns:
                fig4, ax4 = plt.subplots(figsize=(7,3))
                work['pub_year'].value_counts().sort_index().plot(kind='bar', ax=ax4)
                ax4.set_title('Jumlah Dokumen per Tahun (pub_year)')
                ax4.set_xlabel('Tahun')
                ax4.set_ylabel('Jumlah')
                st.pyplot(fig4)
            else:
                st.info("Kolom 'pub_year' tidak tersedia.")

        with lay2:
            present_year_cols = [c for c in YEAR_COLS if c in work.columns]
            if present_year_cols:
                # Asumsi: angka per kolom tahun adalah count/score; kita tampilkan total per tahun
                yearly_totals = work[present_year_cols].sum().astype(float)
                fig5, ax5 = plt.subplots(figsize=(7,3))
                yearly_totals.plot(kind='bar', ax=ax5)
                ax5.set_title('Total Nilai per Tahun (kolom 2000–2022)')
                ax5.set_xlabel('Tahun')
                ax5.set_ylabel('Total')
                st.pyplot(fig5)
            else:
                st.info("Kolom tahun 2000–2022 tidak ditemukan di data.")

        # =====================
        # EKSPOR TOPIK GLOBAL
        # =====================
        st.subheader("📤 Ekspor Hasil Topik Global")
        topic_df = pd.DataFrame([{"Topik": f"Topik {t['topic']+1}", "Kata Kunci": ', '.join(t['terms'][:10])} for t in topics])
        buf_topic = io.BytesIO()
        with pd.ExcelWriter(buf_topic, engine='openpyxl') as writer:
            topic_df.to_excel(writer, index=False, sheet_name="Topik_Global")
        st.download_button("Unduh Topik Global (Excel)", data=buf_topic.getvalue(), file_name="topik_global.xlsx")


        # =====================
        # TOPIK PER DOSEN – FORMAT KALIMAT PER TOPIK
        # =====================
        st.subheader("👨‍🏫 Topik per Dosen – format kalimat per topik")

        if 'author' not in work.columns:
            st.warning("Kolom 'author' tidak ditemukan di data. Bagian ini memerlukan kolom 'author'.")
        else:
            def topic_keywords_for_author(titles: list, topk: int = 10):
                """
                Menghasilkan kata kunci per topik untuk seorang dosen.
                Caranya: re-ranking kata topik global (H[t]) dengan bobot tf-idf
                yang muncul di kumpulan judul milik dosen tsb (s_term).
                """
                if len(titles) == 0:
                    return [[] for _ in range(n_components)]
                X_dosen = vectorizer.transform([sanitize_text(t) for t in titles])
                s_term = np.asarray(X_dosen.sum(axis=0)).ravel()  # panjang = n_features
                kws_per_topic = []
                for t in range(H.shape[0]):
                    scores = H[t] * s_term  # elemen-wise reweighting per term
                    top_idx = np.argsort(scores)[::-1][:topk]
                    terms = [feature_names[i] for i in top_idx]
                    kws_per_topic.append(terms)
                return kws_per_topic

            # Pilihan dosen (ambil 30 teratas agar praktis)
            top_authors = work['author'].value_counts().head(30).index.tolist()
            selected_author = st.selectbox("Pilih Dosen/Penulis", options=top_authors)

            # Tampilkan hasil untuk dosen yang dipilih
            if selected_author:
                titles_sel = work.loc[work['author'] == selected_author, 'title'].dropna().tolist()
                if len(titles_sel) < 3:
                    st.info("Dosen ini memiliki < 3 judul; hasil mungkin kurang stabil.")
                per_topic_terms = topic_keywords_for_author(titles_sel, topk=10)

                # Format kalimat: "Topik 1: a, b, c, ..."
                for t_idx, terms in enumerate(per_topic_terms, start=1):
                    sentence = f"Topik {t_idx}: {', '.join(terms)}"
                    st.markdown(f"**{sentence}**")

                # Opsional: wordcloud per topik untuk dosen terpilih
                if show_wordcloud:
                    st.markdown("—")
                    st.markdown("**Wordcloud per Topik (dosen terpilih)**")
                    cols_wc = st.columns(2)
                    for t_idx, terms in enumerate(per_topic_terms, start=1):
                        with cols_wc[(t_idx-1) % 2]:
                            wc = WordCloud(width=800, height=250, background_color='white').generate(' '.join(terms))
                            fig, ax = plt.subplots()
                            ax.imshow(wc, interpolation='bilinear')
                            ax.set_title(f"Topik {t_idx}")
                            ax.axis('off')
                            st.pyplot(fig)

            # # Ekspor semua dosen → satu Excel: baris per (dosen, topik)
st.markdown("—")
st.subheader("📤 Ekspor Topik untuk Semua Dosen (tag saja)")
min_docs = st.number_input("Minimal jumlah judul per dosen", min_value=1, value=3, step=1)
if st.button("Bangun & Unduh Excel Semua Dosen"):
    rows = []
    for a, cnt in work['author'].value_counts().items():
        if cnt < min_docs:
            continue
        titles_a = work.loc[work['author'] == a, 'title'].dropna().tolist()
        kw_topics = topic_keywords_for_author(titles_a, topk=10)
        for t_idx, terms in enumerate(kw_topics, start=1):
            rows.append({
                'Nama Dosen': a,
                'Topik': f'Topik {t_idx}',
                'Kalimat': ', '.join(terms)   # hanya tag, tanpa "Topik X:"
            })
    if len(rows) == 0:
        st.warning("Tidak ada dosen yang memenuhi ambang minimal judul.")
    else:
        df_out = pd.DataFrame(rows)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_out.to_excel(writer, index=False, sheet_name='Topik_per_Dosen')
        st.download_button(
            "Unduh 'Topik per Dosen' (Excel)",
            data=buf.getvalue(),
            file_name="topik_per_dosen.xlsx"
        )


        # =====================
        # Rekomendasi penggunaan
        # =====================
        with st.expander("ℹ️ Catatan & Tips"):
            st.markdown(
                """
                - **min_df** yang lebih tinggi akan membuang kata yang jarang, cocok untuk dataset besar.
                - **max_df** yang lebih rendah (mis. 0.85) akan membuang kata terlalu umum.
                - Ubah **n-gram** ke *1-3* untuk menangkap frasa multi-kata ("machine learning", "deep learning", dll.).
                - Grafik **Topic Map** menghubungkan node topik dengan kata-kata kunci teratasnya.
                - Heatmap **Dokumen × Topik** hanyalah subset 100 dokumen agar tetap ringan di browser.
                - Anda dapat menambahkan stopword spesifik domain langsung di variabel `CUSTOM_STOPWORDS`.
                """
            )
else:
    st.info("Silakan upload file terlebih dahulu.")
