"""DataPilot Streamlit UI. Run: python -m streamlit run app.py"""
import hashlib
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
    for hint in hints:
        for col in df.columns:
            if hint in col.lower():
                return col
    return None


# ---------- Data source + reset ----------
with st.sidebar:
    st.header("📁 Data")
    up = st.file_uploader("CSV və ya Excel", type=["csv", "xlsx", "xls"])
    demo = st.selectbox("və ya demo fayl", ["—", "sample_data.csv", "sample_data_dirty.csv"])
    st.markdown("---")
    st.write("🤖 Ollama:", "✅ işləyir" if ai_agent.ollama_available() else "❌ tapılmadı (ehtiyat rejim)")
    st.caption(f"Model: `{ai_agent.MODEL}`")

src = up if up is not None else (demo if demo != "—" else None)
if src is None:
    st.info("Soldan CSV/Excel yüklə və ya demo fayl seç.")
    st.stop()

if up is not None:
    digest = hashlib.sha256(up.getvalue()).hexdigest()
    source_key = ("upload", up.name, digest)
else:
    source_key = ("demo", demo)

if st.session_state.get("src_key") != source_key or "df" not in st.session_state:
    st.session_state.update(
        src_key=source_key,
        df=load_file(src),
        original_df=None,
        history=[],
        log=[],
    )

with st.sidebar:
    if st.button("🔄 Reset Data (orijinal fayla qayıt)", use_container_width=True):
        if up is not None:
            up.seek(0)
        st.session_state.update(df=load_file(src), original_df=None, history=[], log=[])
        st.rerun()

df = st.session_state.df
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Ümumi baxış", "🧹 Data keyfiyyəti", "📈 Qrafiklər", "💬 AI Analitik"]
)

# ---------- Overview ----------
with tab1:
    stats = summary_stats(df)
    rev = find_col(df, "revenue", "gəlir")
    prod = find_col(df, "product", "məhsul")
    cols = st.columns(4)
    cols[0].metric("Sətir", f"{stats['rows']:,}")
    cols[1].metric("Sütun", stats["cols"])
    cols[2].metric("Boş xana", stats["missing_total"])
    if rev and pd.api.types.is_numeric_dtype(df[rev]):
        cols[3].metric(f"Ümumi {rev}", f"{df[rev].sum():,.0f}")
    if rev and prod and pd.api.types.is_numeric_dtype(df[rev]):
        grouped = df.groupby(prod)[rev].sum()
        if not grouped.empty:
            st.success(f"🏆 Ən çox {rev} gətirən {prod}: **{grouped.idxmax()}**")
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Sütun tipləri:", stats["dtypes"])
    if stats["numeric"]:
        st.dataframe(pd.DataFrame(stats["numeric"]).T, use_container_width=True)

# ---------- Data quality ----------
with tab2:
    rep = quality_report(df)
    st.metric("Keyfiyyət balı", f"{rep['score']}/100")
    if rep["issues"]:
        st.dataframe(pd.DataFrame(rep["issues"]), use_container_width=True)
    else:
        st.success("Problem tapılmadı 🎉")

    if st.button("✨ Təhlükəsiz düzəlişləri tətbiq et", disabled=not rep["issues"]):
        if st.session_state.get("original_df") is None:
            st.session_state.original_df = df.copy(deep=True)
        cleaned, log = apply_safe_fixes(df)
        st.session_state.df = cleaned
        st.session_state.log = log
        st.rerun()

    st.divider()
    st.subheader("🤖 AI Cleaning Suggestions — Demo")
    st.info("Bu təkliflər hələ sabit nümunədir, Gemma 4-dən gəlmir. Düymələr datanı dəyişmir.")
    demo_suggestions = [
        {"column": "Product", "issue": "Inconsistent text formatting", "suggestion": "Remove extra spaces and standardize names"},
        {"column": "Date", "issue": "Inconsistent date formats", "suggestion": "Convert dates to YYYY-MM-DD"},
    ]
    st.dataframe(pd.DataFrame(demo_suggestions), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve AI Suggestions"):
            st.info("Demo: AI cleaning funksiyası hələ qoşulmayıb.")
    with c2:
        if st.button("❌ Reject AI Suggestions"):
            st.warning("Demo təklifləri rədd edildi. Data dəyişmədi.")

    if st.session_state.get("original_df") is not None:
        st.divider()
        st.subheader("📊 Before / After Cleaning")
        before = st.session_state.original_df
        after = st.session_state.df
        before_score = quality_report(before)["score"]
        after_score = quality_report(after)["score"]
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Before Quality", f"{before_score}/100")
        sc2.metric("After Quality", f"{after_score}/100")
        sc3.metric("Improvement", f"{after_score - before_score:+d} points")
        left, right = st.columns(2)
        with left:
            st.markdown("### Before Cleaning")
            st.metric("Rows", len(before))
            st.dataframe(before.head(10), use_container_width=True)
        with right:
            st.markdown("### After Cleaning")
            st.metric("Rows", len(after))
            st.dataframe(after.head(10), use_container_width=True)

    if st.session_state.get("log"):
        st.subheader("Edilən dəyişikliklər")
        for line in st.session_state.log:
            st.write("•", line)

    st.download_button(
        "⬇️ Cari datanı CSV kimi yüklə",
        df.to_csv(index=False).encode("utf-8-sig"),
        "datapilot_data.csv", "text/csv"
    )
    st.caption("Boş dəyərlər və şübhəli rəqəmlər avtomatik dəyişdirilmir; qərar istifadəçidədir.")

# ---------- Charts ----------
with tab3:
    rev = find_col(df, "revenue", "gəlir")
    dcol = next((c for c in df.columns if is_date_col(c)), None)
    if not rev or not pd.api.types.is_numeric_dtype(df[rev]):
        st.warning("Qrafik üçün rəqəmli Revenue sütunu lazımdır.")
    else:
        if dcol:
            dates = pd.to_datetime(df[dcol], errors="coerce", format="mixed", dayfirst=True)
            d = df.assign(_date=dates).dropna(subset=["_date"])
            if not d.empty:
                monthly = d.set_index("_date")[rev].resample("MS").sum().reset_index()
                st.plotly_chart(px.line(monthly, x="_date", y=rev, markers=True, title=f"Aylıq {rev}"), use_container_width=True)
        graph_cols = st.columns(3)
        for col, hint in zip(graph_cols, ["product", "category", "region"]):
            group_col = find_col(df, hint)
            if group_col:
                grouped = df.groupby(group_col)[rev].sum().sort_values(ascending=False).reset_index()
                col.plotly_chart(px.bar(grouped, x=group_col, y=rev, title=f"{group_col} üzrə {rev}"), use_container_width=True)

# ---------- AI chat ----------
with tab4:
    st.caption("Məs: «Hansı məhsul daha çox gəlir gətirib?», «Hansı ayda satışlar yüksək olub?»")
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.write(entry["text"])
            if entry.get("table") is not None:
                st.dataframe(entry["table"])
    question = st.chat_input("Sualını yaz...")
    if question:
        st.session_state.history.append({"role": "user", "text": question})
        try:
            numeric = list(df.select_dtypes("number").columns)
            context = {
                "numeric": numeric,
                "categorical": [c for c in df.columns if c not in numeric],
                "dtypes": df.dtypes.astype(str).to_dict(),
                "metric": find_col(df, "revenue", "gəlir") or (numeric[0] if numeric else None),
                "date": next((c for c in df.columns if is_date_col(c)), None),
            }
            op = ai_agent.parse_question(question, list(df.columns), context)
            result = run_operation(df, op)
            answer = ai_agent.explain_result(question, result)
            table = pd.DataFrame(result["table"]) if "table" in result else None
            st.session_state.history.append({
                "role": "assistant", "text": f"{answer}\n\n`op: {op}`", "table": table
            })
        except Exception as exc:
            st.session_state.history.append({"role": "assistant", "text": f"⚠️ Sualı icra edə bilmədim: {exc}"})
        st.rerun()
