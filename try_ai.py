"""Manual end-to-end check against a running local Ollama model."""
from pathlib import Path

import pandas as pd

import ai_agent
from data_analyzer import run_operation


def main() -> int:
    df = pd.read_csv(Path(__file__).with_name("sample_data.csv"))
    status = ai_agent.ollama_status()
    print(status["message"])
    print("Installed models:", ", ".join(status["models"]) or "none")
    if not status["available"]:
        return 1
    context = {
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "numeric": ["Quantity", "Revenue"],
        "categorical": ["Product", "Category", "Region"],
        "metric": "Revenue", "date": "Date",
        "values": {c: df[c].dropna().unique().tolist()[:12] for c in ["Product", "Category", "Region"]},
    }
    failures = 0
    questions = [
        "Hansı məhsul daha çox gəlir gətirib?", "Hansı ayda satışlar daha yüksək olub?",
        "Which region has the highest total Revenue?", "Bakıda ümumi gəlir nə qədərdir?",
        "Neçə sətir var?", "Orta gəlir nə qədərdir?",
    ]
    for question in questions:
        try:
            operation = ai_agent.parse_question(question, list(df.columns), context)
            result = run_operation(df, operation)
            print(f"\n{question}\nOperation: {operation}\nResult: {result}")
            print("AI explanation:", ai_agent.explain_result(question, result))
        except (ai_agent.AIError, ValueError) as exc:
            failures += 1
            print(f"\nFAILED: {question}\n{exc}")
    try:
        dirty = pd.read_csv(Path(__file__).with_name("sample_data_dirty.csv"))
        print("\nValidated AI cleaning plan:", ai_agent.get_cleaning_plan(dirty))
        print("Plan was not applied. Review and approve it in Streamlit.")
    except (ai_agent.AIError, ValueError) as exc:
        failures += 1
        print("\nFAILED: AI cleaning plan:", exc)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
