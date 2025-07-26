# Project: Graph-Metric-Based KGQA with TransE and LC-QuAD

## Objective

Build a **Graph-Metric-Based** Knowledge Graph Question Answering (KGQA) system by training the **TransE** embedding model on a knowledge graph derived from **DBpedia**, using **LC-QuAD 1.0** as the question set. The goal is to enable the model to answer structured questions by reasoning over embeddings learned from a real knowledge graph.

---

## Method Summary

We use a **graph-metric-based method** — specifically, the **TransE** model — to embed knowledge graph triples (head, relation, tail) into a continuous vector space. The system is not trained directly on natural language questions. Instead:

1. **LC-QuAD 1.0** provides SPARQL queries paired with natural language questions.
2. We run these SPARQL queries on **DBpedia** to extract valid triples.
3. The resulting triples form the **training data for TransE**.
4. After training, TransE can answer questions by reasoning over the learned embeddings using vector math.

---

## Dataset

### 1. **LC-QuAD 1.0**
- Contains 5,000 natural language questions and their corresponding SPARQL queries.
- Used to identify relevant triples from DBpedia.
- Does not provide answers — instead, the SPARQL query is executed to get the answer.
- File format is JSON.

### 2. **DBpedia**
- External RDF-based knowledge graph.
- Queried using SPARQL (via endpoint or local dump) to extract triples referenced in LC-QuAD SPARQL queries.
- Acts as the ground truth KG for training TransE.

---

## Model

### **TransE (Translation Embedding Model)**
- Implemented using this PyTorch repo: https://github.com/mklimasz/TransE-PyTorch
- Learns vector embeddings for entities and relations such that:
  
  `head + relation ≈ tail`

- The training objective is to make true triples score lower (i.e., closer) than false ones.
- Used for scoring candidate answers via distance metrics in embedding space.

---

## Steps

### A. **Data Preprocessing**
1. Parse LC-QuAD JSON.
2. Extract and batch SPARQL queries.
3. Run SPARQL queries against DBpedia to extract triples.
4. Filter and format triples into `(head, relation, tail)` format for TransE.

### B. **Training TransE**
1. Load all extracted triples.
2. Generate negative samples by corrupting heads or tails.
3. Train TransE using margin-based ranking loss.
4. Save entity and relation embeddings.

### C. **Evaluation**
1. Use held-out triples from LC-QuAD queries for testing.
2. Evaluate link prediction: for a given `(head, relation)`, how close is the correct `tail` in the embedding space?
3. (Optional) Use natural language question → SPARQL → answer pipeline to test full QA performance.

---

## Technologies

- Python
- PyTorch
- NetworkX (optional)
- SPARQLWrapper or rdflib for querying DBpedia
- JSON for parsing LC-QuAD
- TransE training repo: https://github.com/mklimasz/TransE-PyTorch

---

## Deliverables

- Cleaned and structured triple dataset extracted using LC-QuAD and DBpedia.
- Trained TransE model (entity and relation embeddings).
- Evaluation script to score answers based on learned embeddings.
- (Optional) Full question-answer pipeline using embeddings for inference.

---

## Notes

- We **do not construct our own KG**; we use DBpedia as the underlying KG.
- TransE does **not process natural language**; it learns from structured triples.
- SPARQL is only used to extract data — not for training the model directly.
- The success of this approach depends on the quality and coverage of the extracted triples.
