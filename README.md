# DataPilot — AI Data Analyst (Ducktoberfest)

CSV/Excel yüklə → keyfiyyət yoxlaması → təhlükəsiz təmizləmə → qrafiklər → təbii dildə sual.
**Prinsip:** rəqəmləri Python hesablayır, Gemma yalnız sualı JSON əməliyyata çevirir və nəticəni izah edir.

## Qurulum
```bash
pip install -r requirements.txt
python generate_sample_data.py      # sample_data.csv + sample_data_dirty.csv
ollama pull gemma3:4b               # sponsorun dediyi Gemma 4 tag-ı ilə əvəz et
set OLLAMA_MODEL=gemma3:4b          # Linux/Mac: export OLLAMA_MODEL=...
streamlit run app.py
pytest -q                           # testlər
```
Ollama işləmirsə tətbiq ehtiyat (açar söz) rejimində işləyir, demo çökmür.

## Fayllar və sahiblər
| Fayl | Üzv | İş |
|---|---|---|
| data_cleaner.py, data_analyzer.py, generate_sample_data.py | 1 | Data mühərriki |
| app.py | 2 | Streamlit UI |
| ai_agent.py | 3 | Ollama/Gemma, prompt-lar |

## Demo ssenarisi
1. `sample_data_dirty.csv` seç → 2. Keyfiyyət tabı: problemlər → 3. Düzəlişləri tətbiq et → 4. Qrafiklər
→ 5. «Hansı məhsul daha çox gəlir gətirib?» → 6. Təmiz CSV yüklə.

## TODO
- [ ] Gemma 4 tag-ını `ai_agent.MODEL`-də yoxla
- [ ] 15-20 sualluq test siyahısı, prompt tənzimi
- [ ] Ekran videosu (ehtiyat)
