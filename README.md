# AC-RAG: Extraction and Summarization for Home-Visit Nursing Records

Research notebooks for the transcript-to-documentation pipeline (information extraction, AC-RAG summarization, and baselines).

## Code availability

The public GitHub repository URL and persistent identifier (for example, a Zenodo DOI) will be added here after the public repository and archival record are created.

The clinical data (audio, transcripts, nurse-written records, patient profiles) contain personal health information and cannot be released. See the Data Availability statement in the manuscript for the access procedure. `sample_data/` contains synthetic placeholders that document the expected file formats.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # never commit keys
```

Data paths in the notebooks default to `data/`. Place your own data there, following the layout in `sample_data/`.

## Layout

| Component | Files |
|---|---|
| Pipelines, prompts, and baselines | `enhanced-RAG_extractionSummarization.ipynb`, `enhanced-RAG_extractionSummarization_v1.ipynb`, `original-RAG_extractionSummarization.ipynb`, `original-RAG_baseline.ipynb`, `multi-stage_extractionSummarization.ipynb`, `singlePipeline_extractionSummarization.ipynb`, `zero-shot_baseline.ipynb`, `fs-gkp (original).ipynb` |
| Evaluation | `clinical_metrics.py`, `evaluation_pipeline.py`, `evaluation_pipeline_multi.py` |
| Configuration | `requirements.txt` |
| Synthetic input examples | `sample_data/` |

## License

No license has been selected. The code remains subject to its authors' copyright until a license is added.
