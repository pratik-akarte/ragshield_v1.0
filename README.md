🛡️ RAGShield


Automated quality, safety, and PII testing pipeline for production RAG systems.

RAGShield runs functional, adversarial, and data-leak tests against your retrieval-augmented generation (RAG) system, enforcing pass/fail gates directly.

'''
flowchart LR
    A[Query] --> B(Hybrid Search: PGVector + BM25)
    B --> C[FlashRank TinyBERT Cross-Encoder]
    C --> D[Gemini LLM]
    D --> E{Test Engine}
    E -->|Quality| F[DeepEval Metrics]
    E -->|Security| G[Adversarial + Presidio PII]
    F & G --> H[Policy Gate: policy.yaml]
    H -->|Pass| I[Exit 0]
    H -->|Fail| J[Exit 1]
'''