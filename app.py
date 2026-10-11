"""DataPilot Streamlit UI. Run: python -m streamlit run app.py"""
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import ai_agent
from data_analyzer import run_operation, summary_stats
from data_cleaner import (
    apply_cleaning_plan, apply_safe_fixes, is_date_col, load_file,
    preview_cleaning_plan, quality_report,
)

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="DataPilot", page_icon="🛩️", layout="wide")
st.title("🛩️ DataPilot — AI Data Analyst")
st.caption("Python hesablayır, Gemma izah edir. Orijinal dataset həmişə saxlanılır.")


def find_col(df, *hints):
    for hint in hints:
        for col in df.columns:
            if hint in str(col).lower():
                return col
    return None


@st.cache_data(ttl=10, show_spinner=False)
def model_status():
    return ai_agent.ollama_status()


def clear_pending():
    for key in ("ai_plan", "ai_preview", "safe_preview"):
        st.session_state.pop(key, None)


def install_data(data, log):
    """Commit an approved copy; expire plans and answers for the previous data."""
    st.session_state.df = data
    st.session_state.log = st.session_state.log + log
    st.session_state.history = []
    st.session_state.data_version += 1
    clear_pending()


def comparison(before, after, preview=False):
    before_score = quality_report(before)["score"]
    after_score = quality_report(after)["score"]
    cols = st.columns(3)
    cols[0].metric("Before Quality", f"{before_score}/100")
    cols[1].metric("Preview Quality" if preview else "After Quality", f"{after_score}/100")
    cols[2].metric("Improvement", f"{after_score - before_score:+d} points")
    left, right = st.columns(2)
    with left:
        st.markdown("**Before Cleaning**")
        st.caption(f"{len(before)} rows")
        st.dataframe(before.head(20), use_container_width=True)
    with right:
        st.markdown("**Preview — not applied**" if preview else "**After Cleaning**")
        st.caption(f"{len(after)} rows")
        st.dataframe(after.head(20), use_container_width=True)


# ---------- Data source + reset ----------
with st.sidebar:
    st.header("📁 Data")
    up = st.file_uploader("CSV və ya Excel", type=["csv", "xlsx", "xls"])
    demo = st.selectbox("və ya demo fayl", ["—", "sample_data.csv", "sample_data_dirty.csv"])
    st.divider()
    status = model_status()
    st.write("🤖 Ollama:", "✅ hazırdır" if status["available"] else "❌ AI əlçatan deyil")
    st.caption(status["message"])
    st.caption(f"Model: `{ai_agent.MODEL}`")
    if st.button("🔌 Recheck Ollama"):
        model_status.clear()
        st.rerun()

src = up if up is not None else (APP_DIR / demo if demo != "—" else None)
if src is None:
    st.info("Soldan CSV/Excel yüklə və ya demo fayl seç.")
    st.stop()

if up is not None:
    source_key = ("upload", up.name, hashlib.sha256(up.getvalue()).hexdigest())
else:
    source_key = ("demo", demo)

if st.session_state.get("src_key") != source_key or "df" not in st.session_state:
    try:
        if up is not None:
            up.seek(0)
        loaded = load_file(src)
        if len(loaded.columns) == 0:
            raise ValueError("Datasetdə sütun yoxdur.")
    except Exception as exc:
        st.error(f"Fayl oxuna bilmədi: {exc}")
        st.stop()
    clear_pending()
    st.session_state.update(
        src_key=source_key, df=loaded.copy(deep=True), original_df=loaded.copy(deep=True),
        history=[], log=[], data_version=st.session_state.get("data_version", 0) + 1,
        original_bytes=up.getvalue() if up is not None else src.read_bytes(),
        original_name=up.name if up is not None else src.name,
    )

with st.sidebar:
    if st.button("🔄 Reset Data (orijinal fayla qayıt)", use_container_width=True):
        clear_pending()
        st.session_state.update(
            df=st.session_state.original_df.copy(deep=True), history=[], log=[],
            data_version=st.session_state.data_version + 1,
        )
        st.rerun()

df = st.session_state.df
version = st.session_state.data_version
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
        grouped = df.groupby(prod)[rev].sum(min_count=1).dropna()
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

    st.subheader("✨ Basic Cleaning — preview and approve")
    st.caption("Qaydalarla təmizləmə: boşluqlar, birmənalı formatlar və dublikatlar. Bu, AI nəticəsi deyil.")
    if st.button("🔎 Preview Basic Fixes", disabled=not rep["issues"]):
        try:
            preview_data, log = apply_safe_fixes(df, approved=True)
            st.session_state.safe_preview = {"preview_data": preview_data, "log": log}
        except Exception as exc:
            st.error(f"Preview alınmadı: {exc}")
    if st.session_state.get("safe_preview") is not None:
        preview = st.session_state.safe_preview
        comparison(df, preview["preview_data"], preview=True)
        st.write("Təklif edilən dəyişikliklər:", preview["log"])
        approved = st.checkbox("I reviewed and approve these basic changes", key=f"approve_safe_{version}")
        if st.button("✅ Apply Basic Fixes", disabled=not approved):
            install_data(preview["preview_data"].copy(deep=True), preview["log"])
            st.rerun()

    st.divider()
    st.subheader("🤖 AI Cleaning Suggestions — Gemma")
    st.caption("Gemma JSON plan təklif edir. Yalnız təsdiqlənmiş Pandas əməliyyatları icra edilir; Python kodu icra edilmir.")
    st.caption("Median ilə doldurmaq təxmindir; dublikatları silmək sətirləri azaldır. Preview-u yoxlayıb təsdiqləyin.")
    if st.button("🤖 Generate AI Cleaning Plan", disabled=not status["available"]):
        clear_pending()
        try:
            with st.spinner("Gemma cleaning plan hazırlayır..."):
                plan = ai_agent.get_cleaning_plan(df)
                preview = preview_cleaning_plan(df, plan)
            st.session_state.ai_plan = plan
            st.session_state.ai_preview = preview
            st.session_state.plan_id = st.session_state.get("plan_id", 0) + 1
        except Exception as exc:
            st.error(f"AI plan alınmadı; data dəyişmədi: {exc}")
    if st.session_state.get("ai_plan") is not None:
        plan = st.session_state.ai_plan
        preview = st.session_state.ai_preview
        st.json(plan)
        if not plan["operations"]:
            st.info("Gemma icazəli təmizləmə əməliyyatı təklif etmədi.")
        else:
            comparison(df, preview["preview_data"], preview=True)
            st.write("Təklif edilən dəyişikliklər:", preview["log"])
            approved = st.checkbox(
                "I reviewed and approve this AI cleaning plan",
                key=f"approve_ai_{version}_{st.session_state.plan_id}",
            )
            left, right = st.columns(2)
            if left.button("✅ Apply AI Cleaning Plan", disabled=not approved):
                try:
                    cleaned, log = apply_cleaning_plan(df, plan, approved=approved)
                    install_data(cleaned, log)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Plan tətbiq edilmədi: {exc}")
            if right.button("❌ Reject AI Cleaning Plan"):
                clear_pending()
                st.rerun()

    if st.session_state.log:
        st.divider()
        st.subheader("📊 Before / After Cleaning")
        comparison(st.session_state.original_df, df)
        st.subheader("Edilən dəyişikliklər")
        for line in st.session_state.log:
            st.write("•", line)
    left, right = st.columns(2)
    left.download_button(
        "⬇️ Cari datanı CSV kimi yüklə", df.to_csv(index=False).encode("utf-8-sig"),
        "datapilot_data.csv", "text/csv",
    )
    right.download_button(
        "⬇️ Original source file", st.session_state.original_bytes,
        st.session_state.original_name,
        "text/csv" if st.session_state.original_name.lower().endswith(".csv") else "application/octet-stream",
    )
    st.caption("Boş dəyərlər, qeyri-müəyyən tarixlər və şübhəli rəqəmlər səssiz dəyişdirilmir.")

# ---------- Charts ----------
with tab3:
    numeric = [c for c in df.select_dtypes("number").columns
               if not pd.api.types.is_bool_dtype(df[c]) and not pd.api.types.is_complex_dtype(df[c])]
    if not numeric:
        st.info("Qrafik üçün ən azı bir rəqəmsal sütun lazımdır.")
    elif df.empty:
        st.info("Dataset boşdur; qrafik üçün sətir yükləyin.")
    else:
        preferred = find_col(df, "revenue", "gəlir")
        metric = st.selectbox("Chart metric", numeric, index=numeric.index(preferred) if preferred in numeric else 0)
        dcol = next((c for c in df.columns if is_date_col(c) or pd.api.types.is_datetime64_any_dtype(df[c])), None)
        if dcol:
            try:
                trend = run_operation(df, {"op": "trend", "metric": metric, "date_col": dcol, "freq": "M"})
                if trend["invalid_dates"]:
                    st.warning(f"Trend excludes {trend['invalid_dates']} missing, invalid or ambiguous dates; source data is preserved.")
                if trend["table"]:
                    st.plotly_chart(px.line(pd.DataFrame(trend["table"]), x="period", y=metric,
                                            markers=True, title=f"Aylıq {metric}"), use_container_width=True)
            except ValueError as exc:
                st.warning(f"Trend alınmadı: {exc}")
        categories = [c for c in df.columns if c not in numeric and c != dcol]
        preferred_groups = [find_col(df, hint) for hint in ("product", "category", "region")]
        groups = list(dict.fromkeys(c for c in preferred_groups + categories if c in categories))[:3]
        for container, group_col in zip(st.columns(max(1, len(groups))), groups):
            grouped = df.groupby(group_col, dropna=False)[metric].sum(min_count=1).sort_values(ascending=False).head(30).reset_index()
            container.plotly_chart(px.bar(grouped, x=group_col, y=metric,
                                          title=f"{group_col} üzrə {metric}"), use_container_width=True)
        if not groups:
            st.plotly_chart(px.histogram(df, x=metric, title=f"{metric} distribution"), use_container_width=True)

# ---------- AI chat ----------
with tab4:
    st.caption("Məs: «Hansı məhsul daha çox gəlir gətirib?», «Hansı ayda satışlar yüksək olub?»")
    if not status["available"]:
        st.info("AI Analitik üçün Ollama və seçilmiş Gemma modeli lazımdır. Heç bir demo/fallback cavab AI kimi göstərilmir.")
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.write(entry["text"])
            if entry.get("result") is not None:
                st.json(entry["result"])
            if entry.get("table") is not None:
                st.dataframe(entry["table"], use_container_width=True)
    question = st.chat_input("Sualını yaz...", disabled=not status["available"])
    if question:
        st.session_state.history.append({"role": "user", "text": question})
        try:
            numeric = [c for c in df.select_dtypes("number").columns
                       if not pd.api.types.is_bool_dtype(df[c]) and not pd.api.types.is_complex_dtype(df[c])]
            preferred_metric = find_col(df, "revenue", "gəlir")
            context = {
                "numeric": numeric,
                "categorical": [c for c in df.columns if c not in numeric and not is_date_col(c)],
                "dtypes": df.dtypes.astype(str).to_dict(),
                "values": {c: df[c].dropna().astype(str).unique()[:12].tolist()
                           for c in df.columns if c not in numeric},
                "metric": preferred_metric if preferred_metric in numeric else (numeric[0] if numeric else None),
                "date": next((c for c in df.columns if is_date_col(c)), None),
            }
            with st.spinner("Gemma sualı analiz edir..."):
                op = ai_agent.parse_question(question, list(df.columns), context)
                result = run_operation(df, op)
            try:
                with st.spinner("Gemma nəticəni izah edir..."):
                    answer = ai_agent.explain_result(question, result)
            except Exception as exc:
                answer = f"Python result calculated; AI explanation unavailable: {exc}"
            table = pd.DataFrame(result["table"]) if "table" in result else None
            st.session_state.history.append({
                "role": "assistant", "text": f"{answer}\n\nValidated operation: `{op}`",
                "result": result, "table": table,
            })
        except Exception as exc:
            st.session_state.history.append({"role": "assistant", "text": f"⚠️ AI sualı icra edilmədi: {exc}"})
        st.rerun()
