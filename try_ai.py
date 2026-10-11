"""Sürətli AI yoxlaması: python try_ai.py
Real boru kəmərini (parse_question -> run_operation -> explain_result) sınayır.
Model işləməsə ehtiyat parser işə düşür və bu da göstərilir."""
import time

import pandas as pd

import ai_agent
from data_analyzer import run_operation

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
if ai_agent.ollama_available():
    ai_agent.warm_up()

for q in questions:
    t0 = time.time()
    try:
        op = ai_agent.parse_question(q, columns, ctx)
        result = run_operation(df, op)
        print(f"\n✅ {q}\n   op: {op}")
        print("   izah:", ai_agent.explain_result(q, result))
        print(f"   ({time.time() - t0:.1f} san)")
    except Exception as e:
        print(f"\n❌ {q}\n   xəta: {e}")
