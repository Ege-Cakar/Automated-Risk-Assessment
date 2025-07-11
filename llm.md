# Repository Overview for LLMs

This document gives a high-level map of the repository so that a large-language model (LLM) can quickly reason about the codebase, answer questions, and propose edits without scanning every file.

---
## 1. Project Purpose
A **dual-lobe expert agent** is implemented with **LangGraph** (state machine orchestration over LangChain), OpenAI chat models, and a Chroma vector store.  The agent internally deliberates between two specialised roles:

* **Creative Lobe** – brainstorms scenarios / ideas.
* **Reasoning Lobe** – analyses those scenarios and ultimately produces a `CONCLUDE:` answer.

The demo expert (`SecurityExpert`) evaluates cloud-infrastructure risk queries.

---
## 2. Directory / File Structure
```
├── .env               # OpenAI keys & env settings (NOT committed)
├── .gitignore         # standard Python / venv ignores
├── llm.md             # ← you are here
├── src/
│   ├── main.py                    # demo entry-point
│   ├── custom_code/
│   │   ├── expert.py              # ⚙️ Expert orchestrator class (core logic)
│   │   └── lobe.py                # 🤖 Lobe (Creative / Reasoning) wrapper
│   ├── utils/
│   │   ├── memory.py              # Chroma-backed vector memory helper
│   │   └── schemas.py             # TypedDict state schema for LangGraph
│   └── database/                  # persisted Chroma sqlite store
│       └── chroma.sqlite3
└── venv/               # virtualenv (ignored)
```

---
## 3. Key Components & APIs
### 3.1 `Expert` (`src/custom_code/expert.py`)
* Builds a **LangGraph `StateGraph`** with nodes:
  * `initialize`          – prepares both lobes.
  * `lobe1_respond`       – creative step.
  * `lobe2_respond`       – reasoning step.
  * `extract_conclusion`  – stops when `CONCLUDE:` or max rounds.
* Accepts configs (`keywords`, `temperature`, `tools`) for each lobe.
* Public async methods:
  * `process_message(query:str) -> str` – run the deliberation.
  * `update_keywords`, `add_knowledge`.

### 3.2 `Lobe` (`src/custom_code/lobe.py`)
* Thin wrapper around `ChatOpenAI` with its own temperature.
* On first use, fetches contextual docs from shared vector store using its `keywords` and prepends that to the system message.
* `respond(query, context)` → string (async).

### 3.3 `LobeVectorMemory` (`src/utils/memory.py`)
* Convenience wrapper around **Chroma** + **OpenAIEmbeddings**.
* Async helper `search_by_keywords` and `add` for storing documents.

### 3.4 `main.py`
* Demonstration script: instantiates `SecurityExpert` with two lobes and queries about cloud infrastructure security risks.
* Toggle `DEBUG_INTERNAL_DELIBERATION` to print internal exchanges.

---
## 4. Runtime Flow
1. `main.py` loads env vars, creates `LobeVectorMemory`, `ChatOpenAI`, and `Expert`.
2. `Expert.process_message()` kicks off the LangGraph state machine (async).
3. Lobes alternate until either:
   * `Reasoning` replies containing `CONCLUDE:`
   * or max `max_rounds` (default 3) is reached – forcing a summary.
4. Final conclusion is returned to caller / printed.

---
## 5. Environment & Dependencies
* Python ≥ 3.9
* Key external libs: `langchain-openai`, `langchain-chroma`, `langgraph`, `dotenv`, `chromadb`.
* `.env` must define `OPENAI_API_KEY`.
* Vector DB persisted under `src/database/`.

---
## 6. Extending the System
* **Add new experts** by instantiating `Expert` with different `system_message`, `keywords`, temperatures, or tools.
* **Plug custom tools** – pass LangChain tools list in `lobe*_config`.
* **Store domain knowledge** via `Expert.add_knowledge()` or directly adding to Chroma.

---
## 7. Tips for LLM Code Assistants
* All async; remember to `await` `Expert.process_message`.
* Internal state shape is **`ExpertState`** (`utils/schemas.py`).
* To inspect / modify conversation flow, edit `expert.py` node functions or edge conditions.
* For memory retrieval tweak `LobeVectorMemory.config.k` or pass `top_k` to `query_common_db`.

---
**End of Overview**
