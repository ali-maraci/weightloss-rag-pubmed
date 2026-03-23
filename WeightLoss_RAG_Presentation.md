---
marp: true
theme: default
class: default
paginate: true
backgroundColor: "#f8f9fa"
style: |
  section {
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 22px;
    color: #333;
  }
  h1 { color: #1a5f7a; font-size: 2.2em; margin-top: 0px; margin-bottom: 0.2em; border-bottom: 3px solid #e05e5e; padding-bottom: 10px;}
  h2 { color: #227c9d; font-size: 1.5em; margin-top: 0; }
  h3 { color: #e05e5e; font-size: 1.2em; }
  .highlight { background-color: #fff3cd; padding: 2px 6px; border-radius: 4px; font-weight: bold; }

  /* Flexbox Layouts for Visual Contrast */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

  /* Pros & Cons Boxes */
  .pro-box { background: #e8f5e9; padding: 15px; border-radius: 12px; border-left: 6px solid #4caf50; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
  .con-box { background: #ffebee; padding: 15px; border-radius: 12px; border-left: 6px solid #f44336; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
  .info-box { background: #e3f2fd; padding: 15px; border-radius: 12px; border-left: 6px solid #2196f3; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }

  /* Tables */
  table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }
  th { background-color: #1a5f7a; color: white; padding: 10px; text-align: left; font-size: 1.0em; }
  td { padding: 10px; border-bottom: 1px solid #eee; background: white; font-size: 0.9em; }
  tr:last-child td { border-bottom: none; }

  /* Callouts & Badges */
  .badge { background: #227c9d; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px; }
  .badge-alt { background: #e05e5e; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.7em; text-transform: uppercase; }

  /* Specific Custom Layouts and Adjustments (Replacing Inline Styles) */
  .pro-box-purple { background: #f3e5f5; border-left-color: #9c27b0; }
  .pro-box-orange { background: #fff3e0; border-left-color: #ff9800; }
  .con-box-gray { background: #f4f4f4; border-left-color: #555; }
  .con-box-orange { background: #fff3e0; border-left-color: #ff9800; }

  .text-center { text-align: center; }
  .mt-20 { margin-top: 20px; }
  .mt-15 { margin-top: 15px; }
  .mt-10 { margin-top: 10px; }
  .mt-5 { margin-top: 5px; }
  .mt-0 { margin-top: 0; }
  .mt-neg-5 { margin-top: -5px; }
  .mb-20 { margin-bottom: 20px; }
  .mb-15 { margin-bottom: 15px; }
  .mb-10 { margin-bottom: 10px; }
  .mb-5 { margin-bottom: 5px; }
  .mb-2 { margin-bottom: 2px; }
  .mb-neg-10 { margin-bottom: -10px; }
  .m-0 { margin: 0; }
  .p-10 { padding: 10px; }

  .text-small { font-size: 0.85em; }
  .text-xsmall { font-size: 0.75em; }
  .text-xxsmall { font-size: 0.6em; }

  .line-height-tight { line-height: 1.3; }

  .img-rounded { border-radius: 8px; }
  .img-rounded-lg { border-radius: 12px; }
  .img-shadow { box-shadow: 0 4px 6px rgba(0,0,0,0.1); }

  /* Mock UI Box */
  .mock-ui-container { border: 4px solid #227c9d; border-radius: 12px; height: 280px; display: flex; align-items: center; justify-content: center; background: #e3f2fd; color: #1a5f7a; }
  .mock-ui-divider { border-top: 1px solid #1a5f7a; width: 50%; margin: 10px auto; }

  /* Code and Pre */
  pre { background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 10px; font-size: 0.85em; overflow-x: auto; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); }
---

<!-- _class: lead -->
# WeightLoss RAG 🧬
## An End-to-End RAG System over PubMed for GLP-1 & Weight Loss Research

**Ali Maraci**
**23 March 2026**

---

# System Overview

<div class="mt-20">
  <table>
    <tr>
      <th>System Elements</th>
      <th>Implementation Details</th>
    </tr>
    <tr>
      <td><strong>Domain Focus</strong></td>
      <td>GLP-1 medications & weight loss (semaglutide, tirzepatide, liraglutide, Ozempic, Wegovy, Mounjaro)</td>
    </tr>
    <tr>
      <td><strong>Corpus Size</strong></td>
      <td>~5,000 PubMed papers (abstracts + full-text body chunks)</td>
    </tr>
    <tr>
      <td><strong>Chunking Strategy</strong></td>
      <td>Semantic Section-Level (1000 max-tokens, 200 overlap)</td>
    </tr>
    <tr>
      <td><strong>Embedding Model</strong></td>
      <td><code>text-embedding-3-small</code> (OpenAI)</td>
    </tr>
    <tr>
      <td><strong>Vector Database</strong></td>
      <td>Qdrant Cloud (Hybrid Search: Dense Vectors + Sparse BM25)</td>
    </tr>
    <tr>
      <td><strong>Retrieval Architecture</strong></td>
      <td>4-Stage Pipeline (Two-Stage Retrieval, Reranking, Diversity Control)</td>
    </tr>
    <tr>
      <td><strong>Primary LLM (Generator/Parser)</strong></td>
      <td><code>gpt-4o-mini</code> (OpenAI)</td>
    </tr>
    <tr>
      <td><strong>Evaluation</strong></td>
      <td>LLM-as-a-Judge: <code>Llama-3.3-70b-versatile</code> (Groq)</td>
    </tr>
    <tr>
      <td><strong>Core Python Infrastructure</strong></td>
      <td>FastAPI Backend + React 19 Frontend + Firebase Hosting</td>
    </tr>
  </table>
</div>

---

# Navigating the PubMed Database

*Building a RAG on a medical database introduces specific architectural and data processing considerations compared to standard document retrieval.*

<div class="grid-2">
  <div class="info-box">
    <h3 class="mt-neg-5">🔍 Inherent Characteristics</h3>
    <ul class="mt-neg-5">
      <li><strong>Terminology & Synonym Explosion:</strong> Medical concepts have dense, overlapping vocabularies (e.g., "semaglutide," "GLP-1 receptor agonist," "Ozempic"); missing synonyms = missed evidence.</li>
      <li><strong>Evidence Quality & Conflicts:</strong> Mixed study types (case reports → RCTs) with contradictory findings require ranking and context.</li>
      <li><strong>High-Stakes Hallucination Risk:</strong> Small retrieval gaps can yield unsafe conclusions; strict grounding and citations are essential.</li>
    </ul>
  </div>

  <div class="pro-box pro-box-purple">
    <h3 class="mt-neg-5">🎓 The RAG Learning Opportunity</h3>
    <ul class="mt-neg-5">
      <li><strong>Domain-Aware Retrieval:</strong> Learn how query parsing, synonym normalization, and hybrid search (BM25 + embeddings) improve recall in terminology-dense domains.</li>
      <li><strong>Evidence-Aware Ranking:</strong> Learn to rank by study type, recency, and clinical strength, and enforce structured citation grounding to reduce hallucinations.</li>
      <li><strong>Chunking Strategy:</strong> Optimize chunk size and structure (abstract vs. section-level) to balance context, precision, and citation accuracy.</li>
    </ul>
  </div>
</div>

---

# Data Ingestion & Preprocessing

<div class="info-box mb-20">
  <strong>The Dataset:</strong> Open-access PMC articles retrieved via the <strong>NCBI Entrez API</strong>, covering GLP-1 medications across 6 targeted query groups by era and topic (2010–2026).
</div>

### The "Two-Tier JSON" Strategy
To optimize retrieval efficiency, the ingested data was split into two discrete indices.

<div class="grid-2">
  <div>
    <strong>Tier 1 JSON: Abstract + Metadata</strong>
    <p><em>Utilized for fast, broad semantic filtering to establish relevance.</em></p>
  </div>
  <div>
    <strong>Tier 2 JSON: Full-Text + Metadata</strong>
    <p><em>Utilized for deep-dive paragraph extraction once relevant papers are identified.</em></p>
  </div>
</div>

### ✨ Data Transformation
- **Metadata Preservation:** Maintained key metadata such as *MeSH Keywords*, *Publication Type*, *publication year*, and *journal name* to enable rigid filtering during retrieval.
- **Recursive Indicators:** Stripped non-essential XML tags and applied markdown indicators (`##`) to demarcate section headers (e.g., `## Results`).

---

# Chunking Strategy

<div class="grid-2">
  <div>
    <h3>Semantic Chunking Method</h3>
    <p>Documents were parsed natively using the structural headers (<code>##</code>) established during ingestion.</p>
    <ul>
      <li><strong>Abstracts:</strong> Maintained as a single, holistic chunk.</li>
      <li><strong>Full-Text:</strong> Chunked strictly per section (e.g., Methods vs. Results).</li>
      <li><strong>Length Control:</strong> If a section exceeded the optimal length, a fallback of <strong>1000 max-tokens with a 200-token overlap</strong> was applied.</li>
    </ul>
  </div>
  <div class="info-box">
    <h3>Index Size</h3>
    <p>Each chunk inherited the parent document's metadata, alongside an added <code>section_name</code> field.</p>
    <table>
      <tr>
        <th>Index Layer</th>
        <th>Item Count</th>
      </tr>
      <tr>
        <td><strong>Index A</strong> (Abstracts)</td>
        <td>~5,000 documents</td>
      </tr>
      <tr>
        <td><strong>Index B</strong> (Body Chunks)</td>
        <td>~90,000+ chunks</td>
      </tr>
    </table>
  </div>
</div>

---

# Choosing the Embedding Model

*Translating complex biomedicine into mathematical vector representations.*

| Model Option | Type | Clinical Reasoning | Cost / Accessibility |
| :--- | :--- | :--- | :--- |
| **OpenAI `text-emb-3-small`** | Dense | Offers strong general semantic understanding, though may exhibit slightly lower precision on rare acronyms. | Highly cost-effective and accessible. |
| **PubMedBERT** | Dense | Highly specific domain pre-training for biomedical text. | Open source, but strictly limited by a small 512-token context window. |
| **MedEIR** | Dense | Specialized medical embeddings, supports long context windows. | High compute cost and limited testing on rare or new medical terms. |

<div class="pro-box text-small p-10 mt-5">
  <h3 class="mt-5 mb-5">🏆 The Choice: OpenAI & FastEmbed</h3>
  <p class="m-5">For the scope of this project, <strong>OpenAI</strong> was prioritized for dense vectors given its balance of performance and efficiency. Additionally, <strong>FastEmbed</strong> was utilized for sparse vectors to enable native BM25 keyword search.</p>
</div>

---

# The Vector Database Deployment

Initial phase development commenced with a localized **ChromaDB** instance before transitioning to cloud infrastructure.

<div class="grid-2">
  <div class="pro-box">
    <h3>Advantages of Local DB</h3>
    <ul>
      <li>Zero infrastructure cost and immediate setup.</li>
      <li>Operates entirely within the Python runtime environment.</li>
      <li>Ideal for rapid prototyping and offline algorithmic development.</li>
    </ul>
  </div>
  <div class="con-box">
    <h3>Limitations Encountered</h3>
    <ul>
      <li><strong>No Native Hybrid Search:</strong> Out-of-the-box, ChromaDB lacked combined dense vector and BM25 keyword search functionality.</li>
      <li>Because medical queries require exact keyword matching, a manual two-stage Python process had to be engineered.</li>
    </ul>
  </div>
</div>

---

# Two-Stage Semantic Retrieval Strategy

<div class="text-center">
  <img src="data/presentation/system-architecture.png" alt="System Architecture" height="550px" class="img-rounded">
</div>

---

# Prompt Engineering: Stage 1
*Introducing chat history complicates standard RAG retrieval (e.g., resolving pronouns).*

### The Interceptor (Chat History Parser)
**Model:** `gpt-4o-mini` — *A lightweight, high-speed LLM for rapid conversational pre-processing.*
**Configuration:** `temperature=0.0` (Strict Determinism)

* **Role:** An LLM is tasked with reading the <strong>chat history</strong>, <strong>resolving pronouns</strong>, and extracting <strong>explicit identifiers</strong> from previous AI responses into the new query string.
* **Why it matters:** Vector databases cannot "remember" previous chat context. A query like *"What about its side effects?"* will fail a vector search.
* **Example:**
  * **User:** *"What is the standard dosage?"*
  * **Interceptor ➔** *"What is the standard dosage for semaglutide? PMID: 1234567"*

---

# Prompt Engineering: Stage 2

### The Query Parser (Query Optimization & Metadata Extraction)
**Model:** `gpt-4o-mini`
**Configuration:** `temperature=0.0` (Strict Determinism)

Once the query is contextually resolved, a second lightweight LLM chain prepares it for the Hybrid Search engine. It executes 5 core operations:

1. **Clarify and Standardize:** Converts layperson phrases to clinical terms.
2. **Controlled Synonym Expansion:** Adds high-value synonyms without keyword stuffing.
3. **Chunk-Aware Optimization:** Strips conversational filler to keep the query tight.
4. **Ambiguity Detection:** Requests user clarification if a term is hopelessly ambiguous.
5. **Metadata Extraction:** Isolates structured filters using Pydantic models.
   * **Example:** *"Show me trials from 2024 by Smith"* ➔ Extracted Filter: `{ "year": 2024, "author": "Smith" }`

---

# The 4-Stage Retrieval Pipeline

*To combat information overload and ensure high-precision grounding, the retrieval architecture operates in four distinct phases:*

<div class="grid-2">
  <div class="pro-box pro-box-purple">
    <h3>🔍 1. Abstract Discovery (Index A)</h3>
    <p class="text-small m-0">A hybrid search over full article abstracts to quickly identify the top 50 most relevant candidate papers.</p>
  </div>
  <div class="info-box">
    <h3>🎯 2. Deep Chunk Search (Index B)</h3>
    <p class="text-small m-0">A targeted dense search over the specific body paragraphs of ONLY those 50 candidate papers.</p>
  </div>
  <div class="con-box con-box-gray mt-15">
    <h3>⚖️ 3. Algorithmic Reranking</h3>
    <p class="text-small m-0">Adjusting raw vector scores based on methodology strength (e.g., RCTs), recency, and section importance.</p>
  </div>
  <div class="pro-box mt-15">
    <h3>🛡️ 4. Diversity Control</h3>
    <p class="text-small m-0">Hard-capping chunks per PMID to prevent a single review article from monopolizing the LLM's context window.</p>
  </div>
</div>

---

# Stage 1: Abstract Discovery

*Initial broad context retrieval to identify candidate papers.*

<div class="info-box">
  <h3>🔍 Process Overview</h3>
  <ul>
    <li>Runs a <strong>hybrid search (Dense + Sparse BM25)</strong> across <strong>Index A (Abstracts)</strong>.</li>
    <li>Applies any strict <strong>metadata filters</strong> extracted earlier (e.g., Publication Year, Authors).</li>
    <li>Selects the <strong>top 50 unique PubMed IDs (PMIDs)</strong> that are most conceptually aligned with the query.</li>
  </ul>
</div>

---

# Stage 2: Deep Chunk Search

*Drilling down into the specific body paragraphs of identified candidate papers.*

<div class="pro-box">
  <h3>🎯 Targeted Retrieval</h3>
  <ul>
    <li>Executes a <strong>hybrid search (Dense + Sparse BM25)</strong> across <strong>Index B (Full-Text Body Chunks)</strong>.</li>
    <li>Restricts the search space strictly to the <strong>50 Candidate PMIDs found in Stage 1</strong> using a <code>MatchAny</code> filter.</li>
    <li>Pulls the <strong>top 80 most relevant paragraphs/chunks</strong> for deeper analysis.</li>
  </ul>
</div>

*Note: If no candidates are found, a global fallback search on Index B is automatically triggered.*

---

# Stage 3: Algorithmic Reranking

*Vector similarity is not enough. We manually adjust scores based on clinical quality and relevance.*

| Scoring Criteria | Implementation Details | Score Adjustment |
| :--- | :--- | :--- |
| **Methodology** | Prioritizes highest evidence quality. | Meta-Analysis (+1.0), RCTs (+0.85), Case Reports (+0.5) |
| **Section Bias** | Favors key informational sections over filler. | Results/Conclusion (+1.5), Methods (+1.0), Intro (-0.5) |
| **Recency** | Exponential decay curve based on publish year. | `0.25 * exp(-diff / 8)` |

<div class="info-box mt-15 text-small">
  <strong>Mechanism:</strong> The base vector similarity score from Qdrant is summed with the adjustments above. The final set of chunks is re-sorted based on this new holistic score.
</div>

---

# Stage 4: Diversity Control

*Ensuring the LLM context window represents a synthesis of multiple papers.*

<div class="con-box con-box-orange mb-10">
  <h3>🛡️ The Problem: Monopolization</h3>
  <p>Without intervention, a <strong>single highly-relevant review article</strong> might populate a significant portion of the returned chunks, drowning out novel primary research.</p>
</div>

<div class="pro-box">
  <h3>✅ The Solution: Hard-Capping</h3>
  <ul>
    <li>The system enforces a strict maximum of <strong>5 chunks per individual PMID</strong>.</li>
    <li>Truncates the final curated list of varied chunks to the target return size (e.g., <strong>top 30</strong>) before injection into the prompt.</li>
  </ul>
</div>

---

# Global Fallback Search

*What happens when the 4-stage retrieval pipeline fails to provide the Generator LLM with the answer?*

<div class="info-box mb-10">
  <h3>🚨 The Trigger</h3>
  <p>After the initial Pipeline retrieval, the Generator LLM attempts to answer. If it concludes the evidence is insufficient, it safely refuses to answer and drops an internal flag.</p>
</div>

<div class="pro-box pro-box-purple">
  <h3>🔄 Bypassing the Candidates</h3>
  <ul>
    <li>The system automatically intercepts the failure flag.</li>
    <li>It entirely <strong>bypasses Index A</strong> and executes a broad <strong>Hybrid Global Search</strong> (Dense + BM25) directly across the entirety of <strong>Index B (Body Chunks)</strong>.</li>
    <li>This guarantees that hyper-specific findings missed in a paper's abstract can still be retrieved directly from within a paper's body text.</li>
  </ul>
</div>

---

# The Generator LLM

Following context assembly, a generative model synthesizes the final response.

### The LLM Selection: `gpt-4o-mini`
- It provides rapid inference times, is highly cost-effective, and maintains sophisticated instruction-following capabilities relative to its parameter size.

<div class="grid-2 mt-20">
  <div class="con-box con-box-gray">
    <strong>Temperature Specification (0.1)</strong><br>
    In biomedical contexts, variance and creativity equate to hallucination risk. A near-zero temperature enforces near-deterministic outputs, anchoring the LLM strictly to the provided evidentiary chunks.
  </div>
  <div class="con-box con-box-orange">
    <strong>Strict Grounding Constraint</strong><br>
    The system prompt strictly forbids external knowledge. If the generator is unable to find an answer explicitly grounded in the retrieved documents, it is forced to generate an exact flag: <strong><em>"I couldn't find sufficient evidence in the literature to answer this question."</em></strong>
  </div>
</div>

---

# Evaluation: Dual LLM-as-a-Judge

*Quantitative validation using an autonomous framework where one LLM family (OpenAI) generates outputs and another (Groq) judges them, structurally eliminating self-preference bias.*

<div class="grid-2">
  <div>
    <div class="text-center">
      <img src="data/presentation/evaluation-process.png" alt="Evaluation Framework" height="450" class="img-rounded">
    </div>
  </div>
  <div>
    <h3>The Asynchronous Framework</h3>
    <ul>
      <li><strong>Step 1 (Generation):</strong> An LLM (<code>gpt-4o-mini</code>) extracts ground-truth facts from chunks and generates structurally diverse questions into a static test bank JSON.</li>
      <li><strong>Step 2 (Execution):</strong> The RAG pipeline retrieves context and generates an answer without knowledge of the ground truth.</li>
      <li><strong>Step 3 (Judging):</strong> A separate model family (<code>Llama-3.3-70b-versatile</code> via Groq) scores the final answer against the hidden ground truth, preventing self-preference bias.</li>
    </ul>
  </div>
</div>

---

# Cloud Migration: Entering Production

The system transitioned from a local ChromaDB instance to a managed **Qdrant Cloud** database.

| Vector DB Option | Advantages | Disadvantages | Verdict |
| :--- | :--- | :--- | :--- |
| **Qdrant Cloud** | Generous free tier and native support for hybrid (Dense + BM25) search out of the box. | Minor integration adjustments required for LangChain compatibility. | 🏆 **Selected.** Ideal free tier for experimental projects while supporting essential hybrid search capabilities. |
| **Pinecone** | Industry standard, robust APIs. | Proprietary nature and restrictive free-tier limitations. | Not prioritised for this phase. |

*This migration significantly <strong>reduced inference latency</strong> and <strong>decoupled the data layer</strong> from the application compute environment.*

---

# Frontend & Deployment

<div class="grid-2">
  <div>
    <h3>Deployment Infrastructure</h3>
    <ul>
      <li><strong>Frontend:</strong> React 19 + Vite + MUI v7 chat interface with real-time SSE token streaming and inline PubMed citation links.</li>
      <li><strong>Backend:</strong> FastAPI REST endpoint with streaming response support.</li>
      <li><strong>Hosting:</strong> Firebase Hosting (frontend) with the backend designed for containerized deployment via Docker.</li>
    </ul>
  </div>
  <div class="text-center">
    <img src="data/presentation/aura-homepage.jpg" alt="WeightLoss RAG UI" height="530" class="img-rounded-lg img-shadow">
  </div>
</div>

---

# Next Steps and Potential Improvements

<div class="pro-box pro-box-purple">
  <h3>⚙️ Evaluation & Optimization</h3>
  <ul>
    <li><strong>Define Metrics:</strong> Establish component and system-level evaluation metrics (e.g., latency, throughput, memory usage, faithfulness, response relevancy).</li>
    <li><strong>Integrate Phoenix:</strong> Add Arize Phoenix in order to perform detailed tracing, optimization, testing, and continuous evaluation.</li>
    <li><strong>Component Optimization:</strong> Optimize chunking strategy, embedding model, query parser, index A & B retrieval, reranker, diversity control, and choice of generator LLM.</li>
    <li><strong>System Optimization:</strong> Optimize the system architecture balancing <strong>cost</strong> vs <strong>latency</strong> vs <strong>accuracy</strong>.</li>
    <li><strong>Scale Up Corpus:</strong> Expand the corpus beyond 5,000 papers and include additional GLP-1 sub-topics (e.g., paediatric use, drug interactions, long-term outcomes).</li>
  </ul>
</div>

---

<!-- _class: lead -->
# Thank You! 🚀
