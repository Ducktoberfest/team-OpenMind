# DataPilot — AI Data Analyst

DataPilot loads CSV or Excel data, reports quality issues, previews approved cleaning, and builds Plotly charts. The AI Analyst asks a local Gemma model through Ollama for a JSON operation; Pandas computes the answer. AI cleaning plans use validated, allowlisted operations and require explicit approval before application.

## Run locally

Use Python 3.11 or newer. Python 3.12 is the tested cloud runtime. Install [Ollama](https://ollama.com/download) separately for AI features.

Linux / macOS:

```bash
cd team-OpenMind
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=gemma4:e2b
python -m streamlit run app.py
```

Windows PowerShell:

```powershell
cd team-OpenMind
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OLLAMA_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "gemma4:e2b"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. Both sample CSVs are included; choose one in the sidebar or upload your own CSV / `.xlsx` file. CSV files must use UTF-8 encoding; comma, semicolon, tab, and pipe separators are detected. Legacy `.xls` support requires the optional `xlrd` package.

## Start and verify Gemma through Ollama

If Ollama is not already running, start it in a separate terminal:

```bash
ollama serve
```

Check the exact installed model names:

```bash
ollama list
```

The application's default model is `gemma4:e2b`. Pull that tag if it is available from your Ollama installation, then verify that it responds:

```bash
ollama pull gemma4:e2b
ollama run gemma4:e2b "Reply with OK."
```

If your local Gemma 4 installation uses a different tag, set `OLLAMA_MODEL` to that exact name before starting Streamlit. A model tag is not interchangeable with a product name. If a pull reports that the manifest does not exist, use the tag documented by your model provider and shown in `ollama list`.

`OLLAMA_URL` defaults to `http://localhost:11434`. `OLLAMA_TIMEOUT` controls the inference timeout in seconds; CPU inference can be slow. Restart Streamlit after changing these variables. The sidebar checks that the selected model is installed. Without a working model, data loading, quality checks, approved basic fixes, and charts remain available; the sidebar explains the AI connection problem and AI requests are disabled. There are no AI fallback or demo answers.

With Ollama running and the same environment variables set, run the live smoke test:

```bash
python try_ai.py
```

On Windows use `.\.venv\Scripts\python.exe try_ai.py`. This sends sample questions and a cleaning-plan request to your model, prints validated results, and exits with a nonzero status on failure. It does not apply the cleaning plan.

## Hackathon demo

1. Choose `sample_data_dirty.csv` and review the missing values, duplicate rows, whitespace, inconsistent names, suspicious numbers, and date formatting issues.
2. Review the basic cleaning operations and approve them. Missing values, case variants, ambiguous dates, suspicious numbers, and identifiers are preserved unless you explicitly approve a suitable operation.
3. Generate an AI cleaning plan. Inspect its JSON and preview, then approve or reject it. Rejection leaves the dataset unchanged.
4. Compare the original and cleaned data, row counts, and quality scores. Export the cleaned CSV or reset to the original dataset.
5. Open the charts and ask the AI Analyst questions such as “Which product has the highest Revenue?” or “What is the total Revenue?”

Numeric dates such as `01/02/2025` have uncertain day/month order. Trends exclude those values and report the excluded count. Resolve the intended dates in the source file with your team's domain knowledge, save them as `YYYY-MM-DD`, and upload that file again. Quality scores summarize detected issues and do not establish that a dataset is correct.

## Test

Linux / macOS, with the virtual environment activated:

```bash
python -m pytest -q
python -m compileall -q app.py ai_agent.py data_analyzer.py data_cleaner.py generate_sample_data.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app.py ai_agent.py data_analyzer.py data_cleaner.py generate_sample_data.py
```

The automated tests use mocked Ollama responses to verify AI integration and rejection of unsafe or invalid operations. A successful suite does not establish live Gemma inference. On the demo machine, manually verify the installed Gemma tag, a generated cleaning plan, approval/rejection, and an AI Analyst question. Check that the result matches the computed table and that the original data remains downloadable.

## Safety and project files

No AI-generated Python is executed. Both cleaning and analysis validate JSON against operation allowlists, check column names and parameter types, and operate on copies. Cleaning preview is separate from approval. Data resets preserve the source dataset.

| File | Purpose |
|---|---|
| `app.py` | Streamlit upload, quality review, approvals, charts, and AI chat |
| `data_cleaner.py` | File loading, quality checks, validated cleaning and previews |
| `data_analyzer.py` | Summary statistics and validated Pandas analysis |
| `ai_agent.py` | Ollama model checks, JSON plans, question parsing, and explanations |
| `generate_sample_data.py` | Optional sample generation; overwrites both bundled sample CSVs when run |
| `try_ai.py` | Manual smoke test against a live Ollama model; no cleaning is applied |
| `tests/` | Automated regression and integration tests |

The original dataset stays in session memory; export it before closing the session if you need a separate copy. Uploaded data summaries and sample values are sent to the Ollama server configured by `OLLAMA_URL` for AI requests.
