"""Üzv 2: Streamlit interfeysi.  İşə sal:  streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

import ai_agent
from data_analyzer import run_operation, summary_stats
from data_cleaner import apply_safe_fixes, is_date_col, load_file, quality_report

st.set_page_config(page_title="DataPilot", page_icon="🛩️", layout="wide")
st.title("🛩️ DataPilot — AI Data Analyst")
st.caption("Python hesablayır, Gemma izah edir.")


def find_col(df, *hints):
    for h in hints:
        for c in df.columns:
            if h in c.lower():
                return c
    return None


# ---------- Yükləmə ----------
with st.sidebar:
    st.header("📁 Data")
    up = st.file_uploader("CSV və ya Excel", type=["csv", "xlsx", "xls"])
    demo = st.selectbox("və ya demo fayl", ["—", "sample_data.csv", "sample_data_dirty.csv"])
    st.markdown("---")
    st.write("🤖 Ollama:", "✅ işləyir" if ai_agent.ollama_available() else "❌ tapılmadı (ehtiyat rejim)")
    st.caption(f"Model: `{ai_agent.MODEL}`")

src = up if up else (demo if demo != "—" else None)
if src is not None:
    key = getattr(src, "name", src)
    if st.session_state.get("src_key") != key:
        st.session_state.update(src_key=key, df=load_file(src), history=[], log=[])
if "df" not in st.session_state:
    st.info("Soldan fayl yüklə və ya demo fayl seç.")
    st.stop()

df = st.session_state.df
tab1, tab2, tab3, tab4 = st.tabs(["📊 Ümumi baxış", "🧹 Data keyfiyyəti", "📈 Qrafiklər", "💬 AI Analitik"])

# ---------- Ümumi baxış ----------
with tab1:
    s = summary_stats(df)
    rev = find_col(df, "revenue", "gəlir")
    prod = find_col(df, "product", "məhsul")
    c = st.columns(4)
    c[0].metric("Sətir", f"{s['rows']:,}")
    c[1].metric("Sütun", s["cols"])
    c[2].metric("Boş xana", s["missing_total"])
    if rev and pd.api.types.is_numeric_dtype(df[rev]):
        c[3].metric(f"Ümumi {rev}", f"{df[rev].sum():,.0f}")
    if rev and prod and pd.api.types.is_numeric_dtype(df[rev]):
        top = df.groupby(prod)[rev].sum().idxmax()
        st.success(f"🏆 Ən çox {rev} gətirən {prod}: **{top}**")
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Sütun tipləri:", s["dtypes"])
    if s["numeric"]:
        st.dataframe(pd.DataFrame(s["numeric"]).T, use_container_width=True)

# ---------- Keyfiyyət ----------
with tab2:
    rep = quality_report(df)
    st.metric("Keyfiyyət balı", f"{rep['score']}/100")
    if rep["issues"]:
        st.dataframe(pd.DataFrame(rep["issues"]), use_container_width=True)
    else:
        st.success("Problem tapılmadı 🎉")
    if st.button("✨ Təhlükəsiz düzəlişləri tətbiq et", disabled=not rep["issues"]):
        st.session_state.df, st.session_state.log = apply_safe_fixes(df)
        st.rerun()
    if st.session_state.get("log"):
        st.subheader("Edilən dəyişikliklər")
        for line in st.session_state.log:
            st.write("•", line)
    st.download_button("⬇️ Təmiz CSV yüklə", df.to_csv(index=False).encode("utf-8"),
                       "cleaned_data.csv", "text/csv")
    st.caption("Boş dəyərlər və şübhəli rəqəmlər avtomatik dəyişdirilmir, qərar istifadəçidədir.")

# ---------- Qrafiklər ----------
with tab3:
    rev = find_col(df, "revenue", "gəlir")
    dcol = next((c for c in df.columns if is_date_col(c)), None)
    if not rev or not pd.api.types.is_numeric_dtype(df[rev]):
        st.warning("Qrafik üçün rəqəmli Revenue sütunu lazımdır (əvvəlcə təmizləməni tətbiq et).")
    else:
        if dcol:
            d = df.assign(_d=pd.to_datetime(df[dcol], errors="coerce", format="mixed")).dropna(subset=["_d"])
            m = d.set_index("_d")[rev].resample("MS").sum().reset_index()
            st.plotly_chart(px.line(m, x="_d", y=rev, markers=True, title=f"Aylıq {rev}"), use_container_width=True)
        cols = st.columns(3)
        for col, hint in zip(cols, [("product",), ("category",), ("region",)]):
            g = find_col(df, *hint)
            if g:
                t = df.groupby(g)[rev].sum().sort_values(ascending=False).reset_index()
                col.plotly_chart(px.bar(t, x=g, y=rev, title=f"{g} üzrə {rev}"), use_container_width=True)

# ---------- AI ----------
with tab4:
    st.caption("Məs: «Hansı məhsul daha çox gəlir gətirib?», «Hansı ayda satışlar yüksək olub?»")
    for h in st.session_state.history:
        with st.chat_message(h["role"]):
            st.write(h["text"])
            if h.get("table") is not None:
                st.dataframe(h["table"])
    q = st.chat_input("Sualını yaz...")
    if q:
        st.session_state.history.append({"role": "user", "text": q})
        try:
            num = list(df.select_dtypes("number").columns)
            op = ai_agent.parse_question(q, list(df.columns), num)
            result = run_operation(df, op)
            answer = ai_agent.explain_result(q, result)
            table = pd.DataFrame(result["table"]) if "table" in result else None
            st.session_state.history.append({"role": "assistant", "text": answer, "table": table})
            st.session_state.history[-1]["text"] += f"\n\n`op: {op}`"
        except Exception as e:
            st.session_state.history.append({"role": "assistant", "text": f"⚠️ Sualı icra edə bilmədim: {e}"})
        st.rerun()
