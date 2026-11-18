# VibeArchitect

VibeArchitect is a Streamlit prototype that drafts software architecture proposals from short requirements. It retrieves relevant passages from an index of public GitHub documentation and uses them to inform the draft.

## Workflow

1. PyGithub fetches repository documentation. The ingestion script splits Markdown by headings and text length, embeds it with FastEmbed, and persists the index in Chroma.
2. A Groq-hosted model expands the request into a hypothetical reference document (HyDE). Chroma retrieves the five closest passages.
3. LangGraph runs a draft and structured critique. Its routing logic can request another draft, subject to a revision limit.
4. Streamlit shows the resulting proposal and renders Mermaid blocks when present.

| File | Role |
|---|---|
| `data_pipeline.py` | Fetch, chunk, embed, and index documents |
| `agent_core.py` | Retrieval, generation, critique, and routing |
| `app.py` | Streamlit interface |
| `evaluate.py` | Ragas evaluation script |

## Setup and usage

From the repository root, create a Python virtual environment.

```sh
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS or Linux or `.\.venv\Scripts\Activate.ps1` in PowerShell. Then run `python -m pip install -r requirements.txt`.

Create a local `.env`. Set `GITHUB_TOKEN` for GitHub ingestion and `GROQ_API_KEY` for generation. Run ingestion before starting the interface. Ingestion needs network access, and the embedding model may download on first use.

```sh
python data_pipeline.py
python -m streamlit run app.py
```

Enter a short architecture request, such as “Design a shared study-room booking app with conflict checking.” The index is stored in `chroma_db/`.

## Evaluation and limitations

Run `python evaluate.py` with the index and provider credentials configured to measure faithfulness and answer relevance on the included prompts. Results depend on the corpus, model, and prompts. Model critique is an additional review step. Generated designs and diagrams still need human review.
