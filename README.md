# DataPilot — AI Data Analyst (Ducktoberfest)

CSV/Excel yüklə → keyfiyyət yoxlaması → təhlükəsiz təmizləmə → qrafiklər → təbii dildə sual.
**Prinsip:** rəqəmləri Python hesablayır, Gemma yalnız sualı JSON əməliyyata çevirir və nəticəni izah edir.
Gemma xam sətirləri görmür; yalnız sütun adlarını və hesablanmış yekun nəticəni görür. Hamısı lokal işləyir.

## Qurulum

```
pip install -r requirements.txt        # testlər üçün: pip install -r requirements-dev.txt
python generate_sample_data.py         # sample_data.csv + sample_data_dirty.csv
ollama pull gemma4:e2b
streamlit run app.py
pytest -q                              # testlər
```

Başqa model istifadə etmək üçün: `set OLLAMA_MODEL=...` (Linux/Mac: `export OLLAMA_MODEL=...`).
İxtiyari: `OLLAMA_PARSE_TIMEOUT` (default 60 san), `OLLAMA_EXPLAIN_TIMEOUT` (default 90 san).

Ollama işləmirsə və ya model yüklənməyibsə tətbiq ehtiyat (açar söz) rejimində işləyir, demo çökmür.
Sidebar-da "model hazırdır" yazısı həm serveri, həm modelin yüklənməsini yoxlayır.

## Fayllar və sahiblər

| Fayl | Üzv | İş |
| ---- | --- | -- |
| data_cleaner.py, data_analyzer.py, generate_sample_data.py | 1 | Data mühərriki |
| app.py | 2 | Streamlit UI |
| ai_agent.py | 3 | Ollama/Gemma, prompt-lar |

## Sütunlar

Əsas rəqəm, tarix və qruplama sütunları tipinə və adına görə avtomatik təyin olunur;
sidebar-dakı "Sütunlar" bölməsindən istənilən vaxt dəyişmək olar.

## Demo ssenarisi

1. `sample_data_dirty.csv` seç → 2. Keyfiyyət tabı: problemlər → 3. Düzəlişləri tətbiq et → 4. Qrafiklər
→ 5. «Hansı məhsul daha çox gəlir gətirib?» → 6. Təmiz CSV yüklə.

## TODO

- [ ] 15-20 sualluq test siyahısı, prompt tənzimi
- [ ] Ekran videosu (ehtiyat)
