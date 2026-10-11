# Hackathon validation

Validated on Python 3.12.14 with Streamlit 1.65.0, Pandas 3.0.6, Plotly 7.1.0 and NumPy 2.5.4. The project was corrected in place from main at commit `95a70febd0e79110d6c96a3dfc3b7cd9eda07827`. The validated fixes are delivered on `codex-datapilot-fix` for review; main is unchanged and no pull request has been created.

## Verified

- Automated suite: **142 passed**, including 16 Streamlit application cases. No tests were skipped or expected to fail.
- Python compilation and dependency consistency checks passed.
- Streamlit starts and its local health endpoint returns `ok`.
- Both bundled CSVs load; the dirty sample renders four Plotly charts.
- CSV and XLSX parsing, quality detection, cleaning preview/approval/rejection, reset, exports and stale-plan invalidation are covered by application tests. AppTest lacks a browser upload setter, so upload tests supply real Streamlit UploadedFile objects at the upload transport while exercising the real loader and application.
- Original source bytes and original DataFrames are preserved. Identifiers keep leading zeroes; literal NA/NULL category names remain intact.
- Dirty sample approved basic cleaning: **515 → 500 rows**, quality **42 → 72**. Missing values, casing differences, ambiguous dates and suspicious numbers remain for review.
- Real Ollama HTTP integration is exercised with clearly identified mocked responses. JSON validation rejects code, unknown actions, invalid columns, malformed responses and unsafe parameters. All cleaning changes require explicit approval.
- Saved cloud install commands were rerun successfully, and startup instructions were saved. Saving the environment draft does not publish it.

## Still requires the team's machine

Ollama and Gemma weights are absent from this cloud machine. The live check `python try_ai.py` exits 1 with an explicit unavailable-service message. No live AI result is claimed, and there are no fallback/demo answers presented as AI.

1. Start Ollama and confirm the exact installed Gemma 4 tag with `ollama list`. Set OLLAMA_MODEL to it before starting Streamlit; the default is gemma4:e2b.
2. Run `python try_ai.py` to check real cleaning JSON and six analyst questions.
3. In a browser, upload a CSV, generate a plan, inspect its preview, reject it, generate again, approve it, compare before/after, download the original and reset.
4. Ask an analyst question and compare its computed JSON/table to the source. AI explanations are prompt constrained but their factual accuracy still needs live review; Python results are displayed separately.

The supported AI cleaning operations are whitespace trimming, numeric median imputation and duplicate removal. Ambiguous dates and inconsistent casing are detected but never silently guessed. Legacy XLS files require optional xlrd; CSV and XLSX work with the provided dependencies. See README.md for Linux/macOS and Windows instructions.
