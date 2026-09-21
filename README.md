# MindBridge

*Mental Health Dialogue Dataset Pipeline*

**MindBridge** is an end-to-end pipeline that collects public mental-health / empathetic-support conversation datasets from Hugging Face, unifies them into a single `(user_message, assistant_message)` format, cleans them **without data leakage**, and exports LLM-ready splits for fine-tuning.

> **Disclaimer:** This project is for research and educational purposes only. Models trained on this data are **not** a substitute for professional mental-health care. See [Ethical Considerations](#ethical-considerations).

---

## Highlights

- **12 Hugging Face datasets** combined into one unified schema
- **Split-first design** — the train/val/test split happens *before* any preprocessing
- **Modular scikit-learn `Pipeline`** (cleaner → language filter → word-count filter → normalizer → dedup)
- **Leakage checks** at two stages (after splitting and after preprocessing)
- **ChatML-style output** ready for supervised fine-tuning
- Supports **English and Arabic** text through the language filter

---

## Pipeline Overview

```
Download (HF) ─► Parse & Unify ─► SPLIT ─► Preprocess (fit on train) ─► Cross-split
 12 datasets       raw dataframe   80/10/10   clean · filter · dedup      leakage removal
                                                                               │
                                        Save CSVs ◄─ LLM formatting ◄─ EDA
```

| Step | Description |
|------|-------------|
| 1. Config | Split ratios, word-count limits, language threshold, system prompt |
| 2. Download | Loads each dataset with `datasets.load_dataset` and saves every split as CSV |
| 3. Parse & Unify | Converts each dataset's native format (ShareGPT, OpenAI messages, ESConv dialogs, plain Q/A) into `user_message` / `assistant_message` pairs |
| 4. Split first | Respects original `test` / `validation` splits when they exist; splits the remaining pool stratified by `source` |
| 5. Preprocess | `BasicCleaner` → `LanguageFilter` → `WordCountFilter` → `TextNormalizer` → `DeduplicatorPerSplit` |
| 6. Leakage removal | Removes any val/test pair that also appears in train (and test pairs that appear in val) |
| 7. EDA | Row counts, source mix, length buckets |
| 8. Format & save | Builds the ChatML-style `text` column and writes CSV files |

---

## Data Sources

| Key | Hugging Face dataset | Rows in final data |
|-----|----------------------|-------------------:|
| `augesc` | `thu-coai/augesc` | 745,097 |
| `prosocial` | `CAiRE/prosocial-dialog-zho_Hans` | 165,036 |
| `empathetic` | `LuangMV97/Empathetic_counseling_Dataset` | 33,397 |
| `mentalchat` | `ShenLab/MentalChat16K` | 15,384 |
| `esconv` | `thu-coai/esconv` | 13,284 |
| `chillies` | `chillies/psychology-conversation` | 6,008 |
| `mpingale` | `mpingale/mental-health-chat-dataset` | 996 |
| `amod_counseling` | `Amod/mental_health_counseling_conversations` | 746 |
| `safespace` | `danlou/safespace-8877-20230920` | 0 |
| `cogstack` | `openchat/cogstack-opengpt-sharegpt` | 0 |
| `allyarc` | `AllyArc/allyarc_oai_format` | 0 |
| `nart100k` | `jerryjalapeno/nart-100k-synthetic` | 0 |

> Four sources returned 0 rows in the last run — see [Known Issues](#known-issues--limitations).

---

## Results (last run)

| Stage | Rows |
|-------|-----:|
| Raw (unified, before split) | 1,387,515 |
| **Final dataset** | **979,948** |
| ├─ Train | 821,338 |
| ├─ Validation | 73,592 |
| └─ Test | 85,018 |

**Text statistics:** average user message ≈ 17.4 words (median 14); average assistant reply ≈ 24.8 words (median 17).

---

## Getting Started

### 1. Install dependencies

```bash
pip install datasets pandas numpy scikit-learn
```

Python 3.9+ is recommended. The notebook was developed and run on Google Colab.

### 2. Run the pipeline

**Notebook:** open `mental_health_pipeline.ipynb` (Colab or Jupyter) and run all cells.

**Or from Python:**

```python
from pipeline import run_pipeline, CONFIG   # if you export the code to pipeline.py

splits, merged = run_pipeline(CONFIG)
```

> Set a `HF_TOKEN` environment variable to avoid Hugging Face rate limits when downloading.

### 3. Configuration

All settings live in the `CONFIG` dictionary:

| Key | Default | Description |
|-----|---------|-------------|
| `train_ratio` / `val_ratio` / `test_ratio` | `0.80` / `0.10` / `0.10` | Target split ratios (must sum to 1.0) |
| `random_seed` | `42` | Reproducibility |
| `min_words_user` / `max_words_user` | `3` / `300` | Word bounds for user messages |
| `min_words_assist` / `max_words_assist` | `3` / `600` | Word bounds for assistant replies |
| `lang_min_ratio` | `0.80` | Minimum fraction of ASCII or Arabic characters in the user message |
| `system_prompt` | *empathetic assistant prompt* | System prompt inserted into every training example |
| `output_prefix` | `llm` | Prefix of the output CSV files |

---

## Output Files

| File | Columns |
|------|---------|
| `llm_train.csv`, `llm_val.csv`, `llm_test.csv` | `text`, `source`, `split` |
| `merged_mental_health_clean.csv` | `user_message`, `assistant_message`, `source`, `text`, `split` |

Each `text` value follows this ChatML-style template:

```
<|system|>
You are an empathetic mental health support assistant. Listen carefully and respond with compassion and understanding.
<|user|>
{user_message}
<|assistant|>
{assistant_message}
```

> If your training framework expects a specific chat template (e.g. `<|im_start|>` tokens or a tokenizer `chat_template`), adapt `format_chatml()` accordingly.

---

## Data Leakage Prevention

1. **Split before preprocessing** — no statistic is ever computed on the full dataset.
2. **Original splits respected** — rows that came from a dataset's own `test` / `validation` split stay there.
3. **Stratified split by `source`** for the remaining pool.
4. **Per-split deduplication** — each split is deduplicated independently.
5. **Cross-split check** — any val/test pair found in train is removed (train is the source of truth), and test pairs found in val are removed.

---

## Project Structure

```
MindBridge/
├── mental_health_pipeline.ipynb   # Full pipeline (rename from Untitled9)
├── README.md
└── requirements.txt               # optional: datasets, pandas, numpy, scikit-learn
```

---

## Known Issues & Limitations

- **Four sources produce 0 rows** (`safespace`, `cogstack`, `allyarc`, `nart100k`). The conversation-parsing helpers don't match how these datasets are stored after the CSV round-trip. Fixing this would add a lot of conversational data.
- **Source imbalance:** `augesc` alone makes up ~76% of the final data and `prosocial` ~17%. Consider re-sampling or capping per source before training.
- **Large drop for some sources** (e.g. `chillies`: ~348K raw rows → ~6K, `amod_counseling`: ~3.5K → ~0.7K) due to the language, word-count, and dedup filters.
- **Split ratios are approximate.** Because original test/val splits are kept and filtering removes different amounts per split, the final ratio is ≈ 84 / 7.5 / 8.7 rather than exactly 80 / 10 / 10.
- **All pipeline steps are currently stateless** (`dynamic_bounds=False`), so "fit on train only" has no effect yet; it becomes meaningful if you enable percentile-based word-count bounds.
- **Synthetic data:** several sources (e.g. AugESC) are LLM-augmented, so replies may not reflect real clinical practice.
- `langdetect` is installed in the notebook but not used; language filtering is character-based.

---

## Ethical Considerations

- The data covers sensitive topics including **trauma, self-harm, and suicide**. Handle it responsibly and do not redistribute it without checking each source's license.
- Models fine-tuned on it should **not** be deployed as a replacement for licensed professionals. Add crisis-escalation behavior and human oversight for any real-world use.
- Always check and comply with the **license and terms of each original dataset** on Hugging Face before using or sharing derived data.

---

## Roadmap

- [ ] Fix parsing for `safespace`, `cogstack`, `allyarc`, `nart100k`
- [ ] Add per-source sampling caps to balance the data
- [ ] Add proper language detection for Arabic/English
- [ ] Export multi-turn conversations, not only single pairs
- [ ] Package the code as a Python module + CLI

---

## Contributing

Issues and pull requests are welcome. Please open an issue first to discuss major changes.

## License

Add your license here (e.g. MIT). Note that the underlying datasets keep their own licenses.

## Acknowledgements

Thanks to the authors of all the datasets listed above and to the Hugging Face team for `datasets`.
