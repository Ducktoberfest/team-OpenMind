import pandas as pd
import ai_agent
from data_analyzer import run_operation, validate_op

df = pd.read_csv("sample_data.csv")
ctx = {
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "numeric": ["Quantity", "Revenue"],
    "categorical": ["Product", "Category", "Region"],
    "metric": "Revenue",
    "date": "Date",
}
columns = list(df.columns)

questions = [
    "Hansı məhsul daha çox gəlir gətirib?",
    "Hansı ayda satışlar daha yüksək olub?",
    "Which region sold the most?",
    "Bakıda ümumi gəlir nə qədərdir?",
    "Neçə sətir var?",
    "Orta gəlir nə qədərdir?",
]

print("Ollama:", ai_agent.ollama_available(), "| Model:", ai_agent.MODEL)
info = "\n".join(f"- {c}: {ctx['dtypes'][c]}" for c in columns)
sysmsg = ai_agent.PARSE_SYSTEM.format(
    columns_info=info, metric="Revenue", date="Date", dim="Product")

for q in questions:
    try:
        raw = ai_agent._chat(sysmsg, q, json_mode=True)
        op = validate_op(ai_agent._extract_json(raw), columns)
        result = run_operation(df, op)
        print(f"\n✅ {q}\n   op: {op}")
        print("   izah:", ai_agent.explain_result(q, result))
    except Exception as e:
        print(f"\n❌ {q}\n   xəta: {e}")