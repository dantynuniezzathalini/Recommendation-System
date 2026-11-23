import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import fpgrowth, association_rules

st.set_page_config(page_title="Rekomendasi Buku", layout="wide")

DATA_DIR = "data"
FILE_RATING = "BX-Book-Ratings.csv"
FILE_BOOKS = "BX-Books.csv"


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
def cari_file(nama, label):
    # cek di folder data/ dulu, kalau tidak ada minta upload
    path = os.path.join(DATA_DIR, nama)
    if os.path.exists(path):
        return path
    return st.sidebar.file_uploader(label, type="csv")


@st.cache_data(show_spinner="Membaca data...")
def baca_data(src_rating, src_books):
    rating = pd.read_csv(src_rating, sep=";", encoding="latin-1")
    books = pd.read_csv(src_books, sep=";", encoding="latin-1",
                        on_bad_lines="skip", low_memory=False)
    return rating, books


@st.cache_data(show_spinner="Membersihkan data...")
def siapkan_data(rating, books, min_user, min_buku):
    rating = rating.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    dist = rating["Book-Rating"].value_counts().sort_index()

    # rating 0 = implisit, tidak dipakai
    eksplisit = rating[rating["Book-Rating"] > 0].copy()
    n_implisit = int((rating["Book-Rating"] == 0).sum())

    books = books.dropna(subset=["Book-Title", "Book-Author"])
    books = books.drop_duplicates(subset="ISBN").copy()
    books["Book-Title"] = books["Book-Title"].str.strip()

    df = eksplisit.merge(books[["ISBN", "Book-Title", "Book-Author"]],
                         on="ISBN", how="inner")
    df = df.groupby(["User-ID", "Book-Title"], as_index=False)["Book-Rating"].mean()

    # filter user dan buku yang ratingnya terlalu sedikit
    for _ in range(3):
        uc = df["User-ID"].value_counts()
        df = df[df["User-ID"].isin(uc[uc >= min_user].index)]
        bc = df["Book-Title"].value_counts()
        df = df[df["Book-Title"].isin(bc[bc >= min_buku].index)]

    info = {"eksplisit": len(eksplisit), "implisit": n_implisit, "dist": dist}
    return df, info


# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------
@st.cache_data
def hitung_populer(df):
    populer = df.groupby("Book-Title").agg(
        jumlah_rating=("Book-Rating", "count"),
        rata_rata=("Book-Rating", "mean")).reset_index()

    C = df["Book-Rating"].mean()
    m = populer["jumlah_rating"].quantile(0.75)
    n = populer["jumlah_rating"]
    populer["skor"] = (n / (n + m)) * populer["rata_rata"] + (m / (n + m)) * C
    return populer.sort_values("skor", ascending=False).reset_index(drop=True)


@st.cache_resource(show_spinner="Membangun model...")
def bangun_model(df):
    pivot = df.pivot_table(index="Book-Title", columns="User-ID",
                           values="Book-Rating").fillna(0)
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(csr_matrix(pivot.values))
    return pivot, model


def rekomendasi(judul, pivot, model, n=5):
    posisi = pivot.index.get_loc(judul)
    jarak, urutan = model.kneighbors(pivot.iloc[[posisi]], n_neighbors=n + 1)

    hasil = []
    for i in range(1, n + 1):
        hasil.append({
            "Judul": pivot.index[urutan[0][i]],
            "Kemiripan": round(1 - jarak[0][i], 3),
        })
    return pd.DataFrame(hasil, index=range(1, n + 1))


@st.cache_data(show_spinner="Menghitung RMSE...")
def evaluasi(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    semua_user = df["User-ID"].unique()
    semua_buku = df["Book-Title"].unique()
    R = (train.pivot_table(index="User-ID", columns="Book-Title", values="Book-Rating")
              .reindex(index=semua_user, columns=semua_buku).fillna(0))

    S = cosine_similarity(R.values.T)
    np.fill_diagonal(S, 0)

    pembilang = R.values @ S
    penyebut = (R.values > 0).astype(float) @ np.abs(S)
    rata_global = train["Book-Rating"].mean()
    prediksi = np.where(penyebut > 0, pembilang / np.maximum(penyebut, 1e-9), rata_global)

    baris = R.index.get_indexer(test["User-ID"])
    kolom = R.columns.get_indexer(test["Book-Title"])
    y_tebak = prediksi[baris, kolom]
    y_asli = test["Book-Rating"].values

    rmse = np.sqrt(np.mean((y_asli - y_tebak) ** 2))
    mae = np.mean(np.abs(y_asli - y_tebak))
    rmse_base = np.sqrt(np.mean((y_asli - rata_global) ** 2))
    return {"train": len(train), "test": len(test), "rmse": rmse, "mae": mae,
            "rmse_base": rmse_base,
            "perbaikan": (rmse_base - rmse) / rmse_base * 100}


@st.cache_data(show_spinner="Mencari aturan asosiasi...")
def cari_aturan(df, min_like, min_support, min_conf):
    liked = df[df["Book-Rating"] >= min_like]
    basket = pd.crosstab(liked["User-ID"], liked["Book-Title"]) > 0
    basket = basket[basket.sum(axis=1) >= 2]

    frequent = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=3)
    if frequent.empty:
        return basket.shape, 0, pd.DataFrame()

    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    rules = rules[rules["confidence"] >= min_conf].sort_values("lift", ascending=False)
    rules["Jika"] = rules["antecedents"].apply(lambda x: ", ".join(x))
    rules["Maka"] = rules["consequents"].apply(lambda x: ", ".join(x))
    return basket.shape, len(frequent), rules


# ---------------------------------------------------------------
# TAMPILAN
# ---------------------------------------------------------------
st.title("Sistem Rekomendasi Buku")
st.caption("Dataset Book-Crossing | item-based collaborative filtering dan association rule")

st.sidebar.header("Data")
f_rating = cari_file(FILE_RATING, "Upload BX-Book-Ratings.csv")
f_books = cari_file(FILE_BOOKS, "Upload BX-Books.csv")

if f_rating is None or f_books is None:
    st.info("Letakkan file BX-Book-Ratings.csv dan BX-Books.csv di folder `data/`, "
            "atau upload lewat sidebar.")
    st.stop()

st.sidebar.header("Filter data")
min_user = st.sidebar.number_input("Minimal rating per user", 1, 50, 10)
min_buku = st.sidebar.number_input("Minimal rating per buku", 1, 50, 10)

rating, books = baca_data(f_rating, f_books)
df, info = siapkan_data(rating, books, min_user, min_buku)

if df.empty:
    st.error("Data kosong setelah difilter, coba turunkan batas minimal rating.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ringkasan data", "Rekomendasi buku", "Evaluasi RMSE", "Association rule"])

# --- tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris data", f"{len(df):,}")
    c2.metric("Jumlah user", f"{df['User-ID'].nunique():,}")
    c3.metric("Jumlah buku", f"{df['Book-Title'].nunique():,}")
    c4.metric("Rating implisit (0)", f"{info['implisit']:,}")

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Distribusi Book-Rating")
        fig, ax = plt.subplots()
        info["dist"].plot(kind="bar", ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
        st.caption("Sebagian besar data bernilai 0 (rating implisit), "
                   "jadi hanya rating 1-10 yang dipakai.")
    with kanan:
        st.subheader("10 buku terpopuler")
        top = hitung_populer(df).head(10).copy()
        top["rata_rata"] = top["rata_rata"].round(2)
        top["skor"] = top["skor"].round(2)
        st.dataframe(top, hide_index=True)
        st.caption("Skor = weighted rating, jadi buku dengan sedikit rating "
                   "tidak langsung ada di urutan atas.")

# --- tab 2
with tab2:
    pivot, model = bangun_model(df)
    st.write(f"Ukuran matriks buku x user: {pivot.shape[0]} x {pivot.shape[1]}")

    # urutkan judul dari yang paling banyak dirating
    urut = df["Book-Title"].value_counts().index.tolist()
    judul = st.selectbox("Pilih buku yang kamu suka", urut)
    n = st.slider("Jumlah rekomendasi", 1, 10, 5)

    if st.button("Cari rekomendasi"):
        st.write(f"Jika kamu menyukai **{judul}**, coba baca ini:")
        st.table(rekomendasi(judul, pivot, model, n))

# --- tab 3
with tab3:
    st.write("Data dibagi 80% train dan 20% test, lalu rating test diprediksi "
             "dari kemiripan antar buku.")
    if st.button("Hitung RMSE"):
        hasil = evaluasi(df)
        a, b, c = st.columns(3)
        a.metric("RMSE model", f"{hasil['rmse']:.3f}")
        b.metric("MAE model", f"{hasil['mae']:.3f}")
        c.metric("RMSE baseline", f"{hasil['rmse_base']:.3f}")
        st.write(f"Train: {hasil['train']:,} | Test: {hasil['test']:,}")
        st.write(f"Perbaikan dibanding baseline (rata-rata global): "
                 f"**{hasil['perbaikan']:.1f}%**")

# --- tab 4
with tab4:
    st.write("Mencari pola buku yang sering disukai bersamaan.")
    k1, k2, k3 = st.columns(3)
    min_like = k1.slider("Rating dianggap suka (>=)", 5, 10, 7)
    min_support = k2.slider("Min support", 0.005, 0.05, 0.01, 0.005, format="%.3f")
    min_conf = k3.slider("Min confidence", 0.1, 0.9, 0.3, 0.05)

    if st.button("Cari aturan"):
        ukuran, n_freq, rules = cari_aturan(df, min_like, min_support, min_conf)
        st.write(f"Jumlah keranjang: {ukuran[0]:,} | Jumlah buku: {ukuran[1]:,} | "
                 f"Itemset sering: {n_freq} | Jumlah aturan: {len(rules)}")

        if len(rules) == 0:
            st.warning("Tidak ada aturan yang memenuhi, coba turunkan support atau confidence.")
        else:
            tampil = rules[["Jika", "Maka", "support", "confidence", "lift"]].head(10).round(3)
            st.dataframe(tampil, hide_index=True)

            fig, ax = plt.subplots()
            sc = ax.scatter(rules["support"], rules["confidence"],
                            c=rules["lift"], cmap="viridis")
            fig.colorbar(sc, label="Lift")
            ax.set_xlabel("Support")
            ax.set_ylabel("Confidence")
            ax.set_title("Association Rules")
            st.pyplot(fig)
