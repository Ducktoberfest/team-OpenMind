"""DataPilot Streamlit UI. Run: python -m streamlit run app.py"""
import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

import ai_agent
from data_analyzer import run_operation, summary_stats
from data_cleaner import apply_safe_fixes, date_columns, load_file, quality_report

st.set_page_config(page_title="DataPilot", page_icon="🛩️", layout="wide")
st.title("🛩️ DataPilot — AI Data Analyst")
st.caption("Python hesablayır, Gemma izah edir.")


def find_col(df, *hints):
    for hint in hints:
        for col in df.columns:
            if hint in col.lower():
                return col
    return None


def parse_dates(series):
    """ISO tarixləri düz, digərlərini gün-birinci (dayfirst) oxuyur; oxunmayanlar NaT olur."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    values = series.astype("string").str.strip()
    iso = values.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    first = pd.to_datetime(values.where(iso), format="%Y-%m-%d", errors="coerce")
    rest = pd.to_datetime(values.where(~iso & values.notna()), format="mixed",
                          dayfirst=True, errors="coerce")
    return first.fillna(rest)


def safe_load(src):
    """Faylı oxuyur; sütun adlarını mətnə çevirir və təkrarları unikallaşdırır."""
    if hasattr(src, "seek"):
        src.seek(0)
    df = load_file(src)
    seen, cols = {}, []
    for c in df.columns:
        name = str(c)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    df.columns = cols
    return df


@st.cache_resource
def _warm_model():
    """Modeli arxa planda yükləyir ki, səhifə bloklanmasın."""
    import threading
    threading.Thread(target=ai_agent.warm_up, daemon=True).start()
    return True


# ---------- Data source + reset ----------
with st.sidebar:
    st.header("📁 Data")
    up = st.file_uploader("CSV və ya Excel", type=["csv", "xlsx", "xls"])
    demo = st.selectbox("və ya demo fayl", ["—", "sample_data.csv", "sample_data_dirty.csv"])
    st.markdown("---")
    ollama_ok = ai_agent.ollama_available()
    st.write("🤖 Ollama:", "✅ model hazırdır" if ollama_ok else "❌ tapılmadı (ehtiyat rejim)")
    st.caption(f"Model: `{ai_agent.MODEL}`")

if ollama_ok:
    _warm_model()

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
    try:
        loaded = safe_load(src)
    except Exception as exc:
        st.error(f"Fayl oxunmadı: {exc}")
        st.stop()
    st.session_state.update(src_key=source_key, df=loaded, original_df=None, history=[], log=[])

with st.sidebar:
    if st.button("🔄 Reset Data (orijinal fayla qayıt)", use_container_width=True):
        try:
            st.session_state.update(df=safe_load(src), original_df=None, history=[], log=[])
        except Exception as exc:
            st.error(f"Fayl oxunmadı: {exc}")
        st.rerun()

df = st.session_state.df

# ---------- Sütun seçimi (avtomatik təyin + istifadəçi dəyişə bilər) ----------
num_cols = list(df.select_dtypes("number").columns)
date_cols = [c for c in date_columns(df) if c not in num_cols]
cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]

with st.sidebar:
    st.markdown("---")
    st.header("🎯 Sütunlar")
    guess_metric = find_col(df, "revenue", "gəlir", "amount", "sales", "satış", "price", "qiymət")
    metric_col = st.selectbox(
        "Əsas rəqəm sütunu", num_cols or ["—"],
        index=num_cols.index(guess_metric) if guess_metric in num_cols else 0)
    if not num_cols:
        metric_col = None
    other_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]
    date_choice = st.selectbox("Tarix sütunu", ["—"] + date_cols + other_cols,
                               index=1 if date_cols else 0)
    date_col = None if date_choice == "—" else date_choice
    default_groups = [c for c in cat_cols if df[c].nunique() <= 30][:3]
    group_cols = st.multiselect("Qruplama sütunları", cat_cols, default=default_groups)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Ümumi baxış", "🧹 Data keyfiyyəti", "📈 Qrafiklər", "💬 AI Analitik"]
)

# ---------- Overview ----------
with tab1:
    try:
        stats = summary_stats(df)
    except Exception as exc:
        st.error(f"Statistika hesablanmadı: {exc}")
        st.stop()
    cols = st.columns(4)
    cols[0].metric("Sətir", f"{stats['rows']:,}")
    cols[1].metric("Sütun", stats["cols"])
    cols[2].metric("Boş xana", stats["missing_total"])
    if metric_col:
        cols[3].metric(f"Ümumi {metric_col}", f"{df[metric_col].sum():,.0f}")
        if group_cols:
            grouped = df.groupby(group_cols[0])[metric_col].sum()
            if not grouped.empty:
                st.success(f"🏆 Ən çox {metric_col} gətirən {group_cols[0]}: **{grouped.idxmax()}**")
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
        "datapilot_data.csv", "text/csv",
    )
    st.caption("Boş dəyərlər və şübhəli rəqəmlər avtomatik dəyişdirilmir; qərar istifadəçidədir.")

# ---------- Charts ----------
with tab3:
    if not metric_col:
        st.warning("Qrafik üçün ən azı bir rəqəmli sütun lazımdır.")
    else:
        if date_col:
            dates = parse_dates(df[date_col])
            d = df.assign(_date=dates).dropna(subset=["_date"])
            if not d.empty:
                monthly = d.set_index("_date")[metric_col].resample("MS").sum().reset_index()
                st.plotly_chart(
                    px.line(monthly, x="_date", y=metric_col, markers=True, title=f"Aylıq {metric_col}"),
                    use_container_width=True)
            else:
                st.info(f"'{date_col}' sütunundan tarix oxunmadı.")
        if group_cols:
            graph_cols = st.columns(min(3, len(group_cols)))
            for col, group_col in zip(graph_cols, group_cols[:3]):
                grouped = (df.groupby(group_col)[metric_col].sum()
                           .sort_values(ascending=False).head(20).reset_index())
                col.plotly_chart(
                    px.bar(grouped, x=group_col, y=metric_col, title=f"{group_col} üzrə {metric_col}"),
                    use_container_width=True)
        elif not date_col:
            st.info("Soldan tarix və ya qruplama sütunu seç.")

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
            with st.spinner("Gemma düşünür..."):
                ordered_cats = group_cols + [c for c in cat_cols if c not in group_cols]
                context = {
                    "numeric": num_cols,
                    "categorical": ordered_cats,
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "metric": metric_col,
                    "date": date_col,
                }
                op = ai_agent.parse_question(question, list(df.columns), context)
                result = run_operation(df, op)
                answer = ai_agent.explain_result(question, result)
            table = pd.DataFrame(result["table"]) if "table" in result else None
            st.session_state.history.append({
                "role": "assistant", "text": f"{answer}\n\n`op: {op}`", "table": table
            })
        except Exception as exc:
            st.session_state.history.append(
                {"role": "assistant", "text": f"⚠️ Sualı icra edə bilmədim: {exc}"})
        st.rerun()
