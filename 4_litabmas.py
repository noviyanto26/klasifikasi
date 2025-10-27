import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
from sklearn.feature_extraction import text
import time

# Stopwords Indonesia manual
stopwords_id = [
    "yang", "dan", "di", "ke", "dari", "pada", "untuk", "dengan",
    "atau", "juga", "dalam", "karena", "sehingga", "adalah", "itu"
]

# Stopwords Inggris bawaan sklearn
stopwords_en = text.ENGLISH_STOP_WORDS

# Gabungan stopwords ID + EN
stopwords_all = list(set(stopwords_id).union(stopwords_en))

# --- Upload file ---
st.title("📊 Analisis Topik Litabmas Dosen")
uploaded_file = st.file_uploader("Upload file Excel Litabmas", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # --- Tab ---
    tab1, tab2, tab3 = st.tabs(["📋 Data Awal", "🌍 Topik Global", "👩‍🏫 Topik Berdasarkan Nama Dosen"])

    # --- Tab 1: Data Awal ---
    with tab1:
        st.subheader("📋 Data Litabmas")
        st.dataframe(df)

    # --- Tab 2: Topik Global ---
    with tab2:
        st.subheader("🌍 Analisis Topik Global")
        judul_data = df['Judul Litabmas'].dropna().tolist()

        if len(judul_data) >= 5:
            n_topics = st.slider("Jumlah Topik", 2, 10, 5)
            try:
                start_time = time.time()

                progress_global = st.progress(0, text="⏳ Sedang menghitung topik global...")
                eta_placeholder = st.empty()

                # Simulasi progress
                steps = 3
                for i in range(steps):
                    time.sleep(0.3)
                    progress_global.progress(int(((i+1)/steps)*100))
                    elapsed = time.time() - start_time
                    eta_placeholder.text(f"⏱️ Estimasi waktu selesai: {elapsed:.2f} detik (perkiraan)")

                vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stopwords_all)
                tfidf = vectorizer.fit_transform(judul_data)

                nmf_model = NMF(n_components=n_topics, random_state=42).fit(tfidf)
                feature_names = vectorizer.get_feature_names_out()

                exec_time = time.time() - start_time
                st.success(f"✅ Analisis selesai dalam {exec_time:.2f} detik")

                for topic_idx, topic in enumerate(nmf_model.components_):
                    top_keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                    sentence = f"Topik {topic_idx+1}: {', '.join(top_keywords)}"
                    st.markdown(f"**{sentence}**")

                    # Wordcloud
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_keywords))
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Terjadi error: {e}")
        else:
            st.warning("Minimal 5 judul litabmas diperlukan untuk analisis global.")

    # --- Tab 3: Topik per Dosen ---
    with tab3:
        st.subheader("👩‍🏫 Analisis Topik per Dosen")
        selected_dosen = st.selectbox("Pilih Nama Dosen", df['Nama Dosen'].unique())

        if selected_dosen:
            judul_dosen = df[df['Nama Dosen'] == selected_dosen]['Judul Litabmas'].dropna().tolist()
            if len(judul_dosen) >= 3:
                try:
                    start_time = time.time()
                    progress_dosen_single = st.progress(0, text=f"⏳ Sedang menghitung topik untuk {selected_dosen}...")
                    eta_placeholder = st.empty()

                    steps = 3
                    for i in range(steps):
                        time.sleep(0.3)
                        progress_dosen_single.progress(int(((i+1)/steps)*100))
                        elapsed = time.time() - start_time
                        eta_placeholder.text(f"⏱️ Estimasi waktu selesai: {elapsed:.2f} detik (perkiraan)")

                    vectorizer = TfidfVectorizer(stop_words=stopwords_all, min_df=1)
                    tfidf_dosen = vectorizer.fit_transform(judul_dosen)
                    nmf_dosen = NMF(n_components=1, random_state=42).fit(tfidf_dosen)
                    top_keywords = [vectorizer.get_feature_names_out()[i] 
                                    for i in nmf_dosen.components_[0].argsort()[:-11:-1]]

                    exec_time = time.time() - start_time
                    st.success(f"✅ Analisis untuk {selected_dosen} selesai dalam {exec_time:.2f} detik")

                    st.markdown(f"**Topik {selected_dosen}: {', '.join(top_keywords)}**")

                    # Wordcloud
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(top_keywords))
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Terjadi error: {e}")
            else:
                st.warning("Minimal 3 judul litabmas diperlukan untuk analisis per dosen.")

        # --- Download Semua Topik Dosen ---
        st.subheader("📥 Download Semua Topik Dosen")
        dosen_topics_all = []
        total_dosen = len(df['Nama Dosen'].unique())
        progress_dosen = st.progress(0, text="⏳ Sedang menghitung topik untuk semua dosen...")
        eta_placeholder = st.empty()

        start_time_all = time.time()

        for idx, dosen in enumerate(df['Nama Dosen'].unique()):
            judul_dosen_all = df[df['Nama Dosen'] == dosen]['Judul Litabmas'].dropna().tolist()
            nidn_dosen = df[df['Nama Dosen'] == dosen]['NIDN'].iloc[0] if not df[df['Nama Dosen'] == dosen]['NIDN'].empty else None

            if len(judul_dosen_all) >= 3:
                try:
                    vectorizer_dosen = TfidfVectorizer(stop_words=stopwords_all, min_df=1, max_df=1.0)
                    tfidf_temp = vectorizer_dosen.fit_transform(judul_dosen_all)
                    nmf_temp = NMF(n_components=1, random_state=42, max_iter=1000).fit(tfidf_temp)
                    top_kata = [vectorizer_dosen.get_feature_names_out()[i] 
                                for i in nmf_temp.components_[0].argsort()[:-11:-1]]
                    dosen_topics_all.append({
                        "Nama Dosen": dosen,
                        "NIDN": nidn_dosen,
                        "Topik": ", ".join(top_kata)
                    })
                except ValueError:
                    continue

            progress_dosen.progress(int(((idx+1)/total_dosen)*100),
                                    text=f"⏳ Proses dosen {idx+1}/{total_dosen} ...")
            elapsed = time.time() - start_time_all
            avg_time = elapsed / (idx+1)
            remaining = avg_time * (total_dosen - (idx+1))
            eta_placeholder.text(f"⏱️ Estimasi selesai dalam {remaining:.2f} detik lagi")

        exec_time_all = time.time() - start_time_all
        st.success(f"✅ Semua topik dosen selesai dihitung dalam {exec_time_all:.2f} detik")

        if dosen_topics_all:
            df_dosen_topics = pd.DataFrame(dosen_topics_all)
            st.dataframe(df_dosen_topics)
            st.download_button(
                label="📥 Download Topik Semua Dosen (Excel)",
                data=df_dosen_topics.to_csv(index=False).encode('utf-8'),
                file_name="Topik_Semua_Dosen.csv",
                mime="text/csv"
            )
