# Software Architecture & Development Specification: Vectorless RAG Implementation

## 1. Project Overview & Architectural Objectives

Our goal is to build a high-precision Retrieval-Augmented Generation (RAG) system that does not rely on traditional vector databases or embedding models. Instead, we are implementing a Vectorless RAG framework using the open-source PageIndex ecosystem.
Unlike standard RAG, which slices documents into rigid chunk sizes and searches by mathematical similarity, this system uses Large Language Models (LLMs) to build a Hierarchical Tree Index (an intelligent table of contents) of our source documents. The system then uses LLM agents to dynamically traverse this tree, navigating exactly like a human researcher looking at an index to find the precise page or section needed to answer a query.
We will power the intelligence of this entire system using the hosted NVIDIA NIM API Catalog, taking advantage of their high-performance, hosted open-weights models.

---

## 2. Technical Stack & Prerequisites

The development team must provision and configure the following stack:

- Core Framework: pageindex (Python orchestration framework).
- Inference Provider: NVIDIA API Catalog (://nvidia.com).
- Base API Client: Standard OpenAI Python SDK (configured to point to NVIDIA’s API base URL).
- Credentials: An active NVIDIA Developer Account with an authenticated API Key injected into the environment variables.

---

## 3. Step-by-Step Development Execution Plan## Step 1: Environment Initialization & API Connectivity

1.  Environment Variable Management: Securely store the NVIDIA API key in the environment configuration.
2.  SDK Alignment: Instantiate the base LLM client wrapper using the standard OpenAI client layout. Override the default endpoint base URL to point directly to the NVIDIA integration gateway.
3.  Connection Validation: Implement a basic health check script that sends a minimal system prompt to NVIDIA’s hosted platform to verify token authorization and network latency.

## Step 2: Document Ingestion & Structural Parsing

1.  Document Pipeline Setup: Create a secure directory pipeline to accept raw target documents (e.g., PDFs, complex text layouts).
2.  Layout Extraction: Pass incoming documents to the PageIndex ingestion layer. Do not apply any text chunking or character overlapping algorithms. The integrity of data tables and structural sequences must be preserved exactly as written.

## Step 3: Hierarchical Tree Construction (The Indexing Phase)

1.  Model Allocation: Assign a structurally highly capable model—such as Meta Llama 3.3 70B Instruct or NVIDIA Nemotron 3.5 Lightning—specifically for this step.
2.  Index Mapping: Program the framework to read the raw document layout. The LLM must systematically parse headers, sub-sections, page boundaries, and thematic shifts.
3.  Tree Serialization: The system will output a logical "Map" or intelligent tree graph of the document. Each node in the tree must contain the section name, a high-level summary of its contents, and its precise structural coordinates (e.g., page ranges or section IDs). Save this map to disk or cache memory.

## Step 4: Reasoning-Driven Tree Traversal (The Query Phase)

1.  Model Allocation: Assign a reasoning-optimized model—such as DeepSeek V4 Pro or Meta Llama 3.1 70B Instruct—to act as the navigational agent.
2.  The Traversal Loop: When a user submits a natural language question, the reasoning LLM must examine the top-level nodes of our Document Tree Map.
3.  Iterative Navigation: The LLM agent evaluates the section summaries and determines which branch is most likely to hold the answer. It iteratively narrows its focus, moving from parent sections down to specific child subsections until it reaches the targeted location.
4.  Error Handling (Dead Ends): The engine must include logic allowing the LLM to "backtrack" up the tree and choose an alternative path if its first selection yields a dead end.

## Step 5: Answer Synthesis & Source Attribution

1.  Targeted Context Extraction: Once the LLM agent isolates the correct sub-section, extract only that specific block of intact text from the source document.
2.  Final Generation: Feed the targeted text block along with the user’s original query to the NVIDIA NIM endpoint to synthesize a final, human-readable answer.
3.  Metadata Metadata Mapping: Capture and format the exact source paths, page numbers, and headers visited during the traversal. Append these as verifiable citations to the end of the user's response.

---

## 4. Operational Requirements & System Constraints

- Context Window Verification: The team must ensure that the structural blocks or sections parsed in Step 2 do not exceed the context window constraints of the selected NVIDIA models.
- Preventing Infinite Loops: Implement a "Maximum Traversal Depth" or iteration counter. If the LLM navigation loop takes too many hops without isolating an answer, it must gracefully time out and return the most relevant high-level section it found.
- Transition to On-Premise (Future-Proofing): Write all client interactions using standard enterprise API structures. This ensures that when the project scales, we can easily change our code configuration from hosted NVIDIA cloud URLs to local NVIDIA NIM Containers running on our own hardware without rewriting the core business logic.
