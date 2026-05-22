import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from io import BytesIO

st.set_page_config(
    page_title="SKU Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-card {
    background: var(--background-color);
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #e9ecef;
    margin-bottom: 8px;
}
.cluster-header { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

CLUSTER_COLORS = ["#27AE60","#F39C12","#3498DB","#E74C3C","#9B59B6","#1ABC9C"]

POLICY_RULES = {
    "Stable / High Demand Products": {
        "ikmal_modeli":"Sürekli Gözden Geçirme (Q,R) — EOQ",
        "emniyet_stogu":"Düşük–Orta (1–2 hafta)",
        "siparis_sikligi":"Haftalık",
        "tahmin":"Hareketli Ortalama / SES",
        "markdown_riski":"Düşük",
        "color":"#27AE60",
    },
    "Volatile / Short Lifecycle Products": {
        "ikmal_modeli":"Periyodik Gözden Geçirme — Sık, Küçük Sipariş",
        "emniyet_stogu":"Yüksek (3–4 hafta)",
        "siparis_sikligi":"2 Haftada bir",
        "tahmin":"Holt-Winters / Geniş PI",
        "markdown_riski":"Yüksek",
        "color":"#E74C3C",
    },
    "Seasonal Products": {
        "ikmal_modeli":"Sezon Öncesi Toplu Alım + Sezon İçi Takviye",
        "emniyet_stogu":"Orta (zirve öncesi birikim)",
        "siparis_sikligi":"Mevsimsel (yılda 2–4×)",
        "tahmin":"STL / Holt-Winters Mevsimsel",
        "markdown_riski":"Orta–Yüksek",
        "color":"#F39C12",
    },
    "Promotion Sensitive Products": {
        "ikmal_modeli":"Etkinlik Bazlı + Temel İkmal",
        "emniyet_stogu":"Orta + Promo öncesi ek stok",
        "siparis_sikligi":"Düzenli + Taktik promo ikmali",
        "tahmin":"Taban × Promo çarpanı",
        "markdown_riski":"Orta",
        "color":"#3498DB",
    },
    "Regular / Moderate Products": {
        "ikmal_modeli":"Periyodik Gözden Geçirme — Standart",
        "emniyet_stogu":"Orta (2 hafta)",
        "siparis_sikligi":"2 haftada bir / Aylık",
        "tahmin":"Üstel Düzeltme (SES/DES)",
        "markdown_riski":"Düşük–Orta",
        "color":"#9B59B6",
    },
}

@st.cache_data
def load_default_data():
    df = pd.read_excel("sku_clustered_results.xlsx")
    if "CV" not in df.columns:
        df["CV"] = df["volatility"] / df["base_demand"].replace(0, np.nan)
    if "unit_profit" not in df.columns:
        df["unit_profit"] = df["price"] - df["cost"]
    if "lifecycle_risk" not in df.columns:
        df["lifecycle_risk"] = 1 / df["lifecycle_days"].replace(0, np.nan)
    if "lead_time_demand" not in df.columns:
        df["lead_time_demand"] = df["base_demand"] * df["supplier_lead_time_days"]
    return df

def compute_pca(df):
    FEATURES = [f for f in ["base_demand","volatility","CV","supplier_lead_time_days",
                              "gross_margin","lifecycle_days","promo_sensitivity",
                              "seasonality_factor","turnover_rate",
                              "lead_time_demand","lifecycle_risk","unit_profit"]
                if f in df.columns]
    X = df[FEATURES].copy()
    X.replace([np.inf,-np.inf], np.nan, inplace=True)
    for col in X.columns:
        X[col].fillna(X[col].median(), inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.explained_variance_ratio_ * 100, FEATURES

def to_excel_download(df):
    output = BytesIO()
    FEATURES = [f for f in ["base_demand","volatility","CV","supplier_lead_time_days",
                              "gross_margin","lifecycle_days","promo_sensitivity",
                              "seasonality_factor","turnover_rate",
                              "lead_time_demand","lifecycle_risk","unit_profit"]
                if f in df.columns]
    summary = df.groupby("cluster_name")[FEATURES].mean().round(4)
    summary["n_skus"] = df.groupby("cluster_name")["cluster_name"].count()
    names = {row["cluster_id"]: row["cluster_name"]
             for _, row in df[["cluster_id","cluster_name"]].drop_duplicates().iterrows()}
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="SKU Listesi", index=False)
        summary.to_excel(writer, sheet_name="Küme Özeti")
        pd.DataFrame([
            {"Küme ID": k, "Küme Adı": v,
             **{key: POLICY_RULES.get(v.split(" (")[0],
                POLICY_RULES["Regular / Moderate Products"])[key]
                for key in ["ikmal_modeli","emniyet_stogu","siparis_sikligi","tahmin","markdown_riski"]}}
            for k, v in names.items()
        ]).to_excel(writer, sheet_name="Politikalar", index=False)
    return output.getvalue()

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    try:
        st.session_state.df = load_default_data()
    except:
        st.session_state.df = None

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 SKU Planner")
    st.caption("Fast Fashion Segmentasyon Sistemi")
    st.divider()
    page = st.radio("Sayfa", [
        "🏠 Dashboard",
        "📂 Veri Yükle",
        "🔍 SKU Explorer",
        "📊 Segmentasyon",
        "📋 Politikalar"
    ], label_visibility="collapsed")
    st.divider()
    if st.session_state.df is not None:
        df = st.session_state.df
        st.caption(f"**{len(df):,}** SKU yüklü")
        st.caption(f"**{df['cluster_id'].nunique()}** küme")
    st.caption("EM Bitirme Projesi · 2025")

df = st.session_state.df

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("📦 SKU Planner")
    st.caption("Fast Fashion SKU Segmentasyon & Tedarik Planlama Sistemi")

    if df is None:
        st.warning("Veri yüklenemedi. **Veri Yükle** sayfasına git.")
    else:
        names_map = df[["cluster_id","cluster_name"]].drop_duplicates().set_index("cluster_id")["cluster_name"].to_dict()
        counts    = df.groupby("cluster_id").size()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Toplam SKU",   f"{len(df):,}")
        c2.metric("Küme Sayısı",  df["cluster_id"].nunique())
        c3.metric("Kategori",     df["category_group"].nunique() if "category_group" in df.columns else "—")
        c4.metric("Ürün Tipi",    df["product_type"].nunique()   if "product_type"   in df.columns else "—")

        st.divider()
        st.subheader("Küme profilleri")
        cols = st.columns(len(names_map))
        for i, (cid, name) in enumerate(sorted(names_map.items())):
            base  = name.split(" (")[0]
            color = POLICY_RULES.get(base, POLICY_RULES["Regular / Moderate Products"])["color"]
            n_sku = int(counts.get(cid, 0))
            pct   = round(n_sku / len(df) * 100, 1)
            pol   = POLICY_RULES.get(base, POLICY_RULES["Regular / Moderate Products"])
            risk  = pol["markdown_riski"]
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid {color}">
                  <div class="cluster-header" style="color:{color}">C{cid}</div>
                  <div style="font-size:12px;margin-bottom:8px;line-height:1.4">{name}</div>
                  <div style="font-size:24px;font-weight:700">{n_sku}</div>
                  <div style="font-size:11px;color:#888">SKU · %{pct}</div>
                  <div style="font-size:11px;margin-top:6px;color:#666">Risk: {risk}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sizes  = [int(counts.get(cid,0)) for cid in sorted(names_map)]
        labels = [f"C{cid}\n{n[:20]}" for cid,n in sorted(names_map.items())]
        colors = [POLICY_RULES.get(n.split(" (")[0], POLICY_RULES["Regular / Moderate Products"])["color"]
                  for _,n in sorted(names_map.items())]
        axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize":8})
        axes[0].set_title("SKU dağılımı", fontsize=11)

        if "category_group" in df.columns:
            cats = df["category_group"].value_counts().head(6)
            axes[1].barh(cats.index, cats.values, color="#3498DB", alpha=0.8)
            axes[1].set_title("Kategori dağılımı (Top 6)", fontsize=11)
            axes[1].set_xlabel("SKU sayısı")
        else:
            axes[1].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Veri Yükle":
    st.title("📂 Veri Yükle")
    st.info("Varsayılan olarak `sku_clustered_results.xlsx` yüklü. Farklı bir dosya yüklemek istersen aşağıyı kullan.")

    uploaded = st.file_uploader("CSV veya Excel yükle", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                df_new = pd.read_excel(uploaded)
            else:
                df_new = pd.read_csv(uploaded)
            if "CV" not in df_new.columns:
                df_new["CV"] = df_new["volatility"] / df_new["base_demand"].replace(0,np.nan)
            if "unit_profit" not in df_new.columns:
                df_new["unit_profit"] = df_new["price"] - df_new["cost"]
            if "lifecycle_risk" not in df_new.columns:
                df_new["lifecycle_risk"] = 1 / df_new["lifecycle_days"].replace(0,np.nan)
            if "lead_time_demand" not in df_new.columns:
                df_new["lead_time_demand"] = df_new["base_demand"] * df_new["supplier_lead_time_days"]
            st.session_state.df = df_new
            st.success(f"✓ {uploaded.name} yüklendi — {len(df_new):,} satır")
            st.dataframe(df_new.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Hata: {e}")

    if df is not None:
        st.divider()
        st.subheader("Mevcut veri önizleme")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{len(df):,} satır · {df.shape[1]} sütun")

# ══════════════════════════════════════════════════════════════════════════════
# SKU EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SKU Explorer":
    st.title("🔍 SKU Explorer")

    if df is None:
        st.warning("Veri yüklenemedi.")
    else:
        names_map = df[["cluster_id","cluster_name"]].drop_duplicates().set_index("cluster_id")["cluster_name"].to_dict()
        c1,c2,c3 = st.columns(3)
        cluster_opts = ["Tümü"] + [f"C{k}: {v}" for k,v in sorted(names_map.items())]
        sel_c  = c1.selectbox("Küme", cluster_opts)
        pt_opts = ["Tümü"] + sorted(df["product_type"].dropna().unique().tolist()) if "product_type" in df.columns else ["Tümü"]
        sel_pt = c2.selectbox("Ürün tipi", pt_opts)
        search = c3.text_input("SKU ara", placeholder="W-CL-TSB-0001")

        dv = df.copy()
        if sel_c != "Tümü":
            cid = int(sel_c.split(":")[0].replace("C","").strip())
            dv  = dv[dv["cluster_id"] == cid]
        if sel_pt != "Tümü" and "product_type" in dv.columns:
            dv = dv[dv["product_type"] == sel_pt]
        if search and "SKU" in dv.columns:
            dv = dv[dv["SKU"].str.contains(search, case=False, na=False)]

        st.caption(f"{len(dv):,} SKU gösteriliyor")
        show = [c for c in ["SKU","cluster_name","product_type","subcategory",
                              "base_demand","CV","lifecycle_days","promo_sensitivity",
                              "gross_margin","unit_profit"] if c in dv.columns]
        st.dataframe(dv[show].reset_index(drop=True), use_container_width=True, height=480)

        st.divider()
        excel_bytes = to_excel_download(dv)
        st.download_button("⬇️ Filtrelenmiş listeyi Excel indir", data=excel_bytes,
                           file_name="sku_filtreli.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTASYON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Segmentasyon":
    st.title("📊 Segmentasyon & PCA")

    if df is None:
        st.warning("Veri yüklenemedi.")
    else:
        names_map = df[["cluster_id","cluster_name"]].drop_duplicates().set_index("cluster_id")["cluster_name"].to_dict()

        with st.spinner("PCA hesaplanıyor..."):
            X_pca, var, features = compute_pca(df)

        st.subheader("PCA projeksiyonu")
        st.caption(f"PC1: %{var[0]:.1f} varyans · PC2: %{var[1]:.1f} varyans · Toplam: %{sum(var):.1f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        for cid, name in sorted(names_map.items()):
            mask  = (df["cluster_id"] == cid).values
            color = POLICY_RULES.get(name.split(" (")[0],
                    POLICY_RULES["Regular / Moderate Products"])["color"]
            ax.scatter(X_pca[mask,0], X_pca[mask,1],
                       c=color, label=f"C{cid}: {name}", alpha=0.7, s=45, edgecolors="none")
        ax.set_xlabel(f"PC1 ({var[0]:.1f}% varyans)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var[1]:.1f}% varyans)", fontsize=11)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_title("SKU Kümeleri — PCA Projeksiyonu", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Küme özet tablosu")
        num_features = [f for f in features if f in df.columns]
        summary = df.groupby("cluster_name")[num_features].mean().round(3)
        summary.insert(0, "n_skus", df.groupby("cluster_name").size())
        st.dataframe(summary, use_container_width=True)

        st.divider()
        if "product_type" in df.columns:
            st.subheader("Küme × Ürün tipi")
            ct = pd.crosstab(df["cluster_name"], df["product_type"])
            st.dataframe(ct, use_container_width=True)

        if "subcategory" in df.columns:
            st.subheader("Küme × Alt kategori (Top 10)")
            ct2 = pd.crosstab(df["cluster_name"], df["subcategory"])
            top_cols = ct2.sum().nlargest(10).index
            st.dataframe(ct2[top_cols], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# POLİTİKALAR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Politikalar":
    st.title("📋 Tedarik & Stok Politikaları")

    if df is None:
        st.warning("Veri yüklenemedi.")
    else:
        names_map = df[["cluster_id","cluster_name"]].drop_duplicates().set_index("cluster_id")["cluster_name"].to_dict()
        counts    = df.groupby("cluster_id").size()

        for cid, name in sorted(names_map.items()):
            base  = name.split(" (")[0]
            pol   = POLICY_RULES.get(base, POLICY_RULES["Regular / Moderate Products"])
            color = pol["color"]
            n_sku = int(counts.get(cid, 0))

            with st.expander(f"C{cid} — {name}  ({n_sku} SKU)", expanded=True):
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**İkmal Modeli**\n\n{pol['ikmal_modeli']}")
                c2.markdown(f"**Emniyet Stoğu**\n\n{pol['emniyet_stogu']}")
                c3.markdown(f"**Sipariş Sıklığı**\n\n{pol['siparis_sikligi']}")
                c1b,c2b = st.columns(2)
                c1b.markdown(f"**Tahmin Yöntemi**\n\n{pol['tahmin']}")
                risk_icon = "🔴" if "Yüksek" in pol["markdown_riski"] else ("🟡" if "Orta" in pol["markdown_riski"] else "🟢")
                c2b.markdown(f"**İndirim Riski**\n\n{risk_icon} {pol['markdown_riski']}")

        st.divider()
        excel_bytes = to_excel_download(df)
        st.download_button("⬇️ Tüm sonuçları Excel indir", data=excel_bytes,
                           file_name="sku_segmentasyon_sonuclari.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
