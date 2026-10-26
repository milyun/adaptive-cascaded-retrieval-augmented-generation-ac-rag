# Adaptive Cascaded Retrieval-Augmented Generation (AC-RAG)

Extraction and Summarization for Home-Visit Nursing Records.

Research notebooks for the transcript-to-documentation pipeline (information extraction, AC-RAG summarization, and baselines).

## Code availability

The source code supporting this work is available at [GitHub](https://github.com/milyun/adaptive-cascaded-retrieval-augmented-generation-ac-rag). The archived v0.1.0 release is available on Zenodo at [doi:10.5281/zenodo.23656597](https://doi.org/10.5281/zenodo.23656597). The repository includes the pipeline and prompt notebooks, evaluation modules, configuration, and synthetic example records; clinical data are not included.

The clinical data (audio, transcripts, nurse-written records, patient profiles) contain personal health information and cannot be released. See the Data Availability statement in the manuscript for the access procedure. `sample_data/` contains synthetic example records that document the expected file formats.

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

## Suggested Citation

Shoumi MN. Adaptive Cascaded Retrieval-Augmented Generation (AC-RAG): initial public source release, v0.1.0. Zenodo. 2026. doi:10.5281/zenodo.23656597.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
