# Traditional RAG vs Vectorless RAG — Build Tasks

- `[x]` **Step 1: Dependencies & Config**
  - `[x]` Install numpy, update requirements.txt
  - `[x]` Update config.py, .env, .env.example

- `[x]` **Step 2: Refactor Vectorless RAG into sub-package**
  - `[x]` Create `services/vectorless_rag/` with __init__, index_service, query_service
  - `[x]` Remove old service files
  - `[x]` Update all imports (routes, etc.)

- `[x]` **Step 3: Traditional RAG Pipeline**
  - `[x]` `services/traditional_rag/__init__.py`
  - `[x]` `services/traditional_rag/chunker.py`
  - `[x]` `services/traditional_rag/embedding_service.py`
  - `[x]` `services/traditional_rag/vector_store.py`
  - `[x]` `services/traditional_rag/retrieval_service.py`

- `[x]` **Step 4: Data Models**
  - `[x]` Add Chunk, TraditionalQueryResult, ComparisonResult to models

- `[x]` **Step 5: Comparison Service**
  - `[x]` `services/compare_service.py`

- `[x]` **Step 6: Routes**
  - `[x]` Update api.py with traditional + compare endpoints
  - `[x]` Update main.py with compare page route

- `[x]` **Step 7: Frontend**
  - `[x]` Create compare.html template
  - `[x]` Update query.html with compare button
  - `[x]` Update style.css with comparison styles
  - `[x]` Update app.js with comparison API calls

- `[x]` **Step 8: Verify**
  - `[x]` Restart server and test
  - `[x]` Unit tests passing (5/5 passed)
  - `[x]` Live UI verified via browser testing
