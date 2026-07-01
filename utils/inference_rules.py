"""
Capability inference rules.

Maps detected evidence to higher-level capabilities.

FIX: the original rule set only boosted retrieval, llm, embeddings,
vector_db, backend, cloud, and data_engineering/streaming. Five
capabilities — production_ml, evaluation, ranking, recommendation,
distributed — had ZERO inference rules, meaning they could only ever
score from direct keyword/phrase matches with no compounding bonus.
This systematically under-scored candidates with genuine production
ML / ranking / evaluation experience relative to candidates who merely
mentioned retrieval or LLM tooling. The JD's "absolutely need" section
is built almost entirely around production_ml, retrieval, vector_db,
and evaluation — so production_ml and evaluation in particular cannot
be left without inference bonuses.

New rules below restore parity for these five capabilities.
"""

INFERENCE_RULES = [

    # -----------------------------
    # Retrieval Systems
    # -----------------------------

    {
        "if_any": {
            "FAISS",
            "Milvus",
            "Pinecone",
            "Qdrant",
            "Weaviate",
        },
        "infer": {
            "retrieval": 3,
            "vector_db": 3,
            "embeddings": 2,
        },
    },

    {
        "if_any": {
            "Sentence Transformers",
            "Embeddings",
            "Vector Search",
            "Semantic Search",
        },
        "infer": {
            "retrieval": 2,
            "embeddings": 3,
        },
    },

    {
        "if_any": {
            "BM25",
            "Learning to Rank",
            "Cross Encoder",
            "Bi Encoder",
            "Reranking",
        },
        "infer": {
            "retrieval": 3,
        },
    },

    # -----------------------------
    # LLM
    # -----------------------------

    {
        "if_any": {
            "LangChain",
            "LlamaIndex",
            "Prompt Engineering",
        },
        "infer": {
            "llm": 3,
        },
    },

    {
        "if_any": {
            "LoRA",
            "QLoRA",
            "PEFT",
        },
        "infer": {
            "llm": 3,
            "fine_tuning": 3,
        },
    },

    {
        "if_any": {
            "RAG",
        },
        "infer": {
            "llm": 2,
            "retrieval": 3,
        },
    },

    # -----------------------------
    # Backend
    # -----------------------------

    {
        "if_any": {
            "FastAPI",
            "Flask",
            "Django",
            "Spring Boot",
        },
        "infer": {
            "backend": 3,
        },
    },

    {
        "if_any": {
            "REST APIs",
            "GraphQL",
            "gRPC",
        },
        "infer": {
            "backend": 2,
        },
    },

    # -----------------------------
    # Cloud
    # -----------------------------

    {
        "if_any": {
            "AWS",
            "Azure",
            "GCP",
        },
        "infer": {
            "cloud": 3,
        },
    },

    {
        "if_any": {
            "Docker",
            "Kubernetes",
        },
        "infer": {
            "cloud": 2,
            "devops": 3,
            # NEW: containerized deployment is a strong production_ml
            # signal — most real ML deployment work involves Docker/K8s.
            "production_ml": 2,
        },
    },

    # -----------------------------
    # Data Engineering
    # -----------------------------

    {
        "if_all": {
            "Kafka",
            "Spark",
        },
        "infer": {
            "data_engineering": 4,
            "streaming": 3,
            # NEW: large-scale streaming pipelines imply distributed
            # systems experience — was previously unrewarded.
            "distributed": 3,
        },
    },

    {
        "if_any": {
            "Airflow",
            "ETL",
            "Data Pipelines",
        },
        "infer": {
            "data_engineering": 3,
        },
    },

    # -----------------------------
    # NEW: Production ML
    # -----------------------------
    # The JD's #1 "absolutely need" — production deployment experience.
    # Previously had zero inference rules; only the 2 explicit skills
    # (MLflow, Kubeflow) and 8 phrases in capability_schema.py could
    # contribute, with no compounding from related signals.

    {
        "if_any": {
            "MLflow",
            "Kubeflow",
        },
        "infer": {
            "production_ml": 3,
        },
    },

    {
        "if_any": {
            "AWS",
            "Azure",
            "GCP",
        },
        "infer": {
            # Cloud experience alone is weak evidence of production ML,
            # but combined with explicit ML/serving terms it's meaningful.
            # Kept small (1) since cloud already has its own rule above.
            "production_ml": 1,
        },
    },

    # -----------------------------
    # NEW: Evaluation
    # -----------------------------
    # JD explicitly requires "hands-on experience designing evaluation
    # frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online
    # correlation, A/B test interpretation." Previously zero inference
    # rules and zero explicit skills (empty set in capability_schema.py),
    # relying entirely on phrase matches.

    {
        "if_any": {
            "Learning to Rank",
            "XGBoost",
            "LambdaMART",
            "LightGBM",
        },
        "infer": {
            "ranking": 3,
            "evaluation": 2,
        },
    },

    {
        # NOTE: capability_detector.py normalizes (lowercases) both the
        # detected_terms keys and these rule terms before matching, so
        # casing here is just for readability — it no longer has to match
        # capability_schema.py's exact casing to fire.
        "if_any": {
            "A/B Testing",       # skill, capability_schema.py
            "Offline Evaluation",  # skill, capability_schema.py
            "ab testing",         # phrase, capability_schema.py
            "a/b testing",        # phrase, capability_schema.py
            "offline benchmark",  # phrase, capability_schema.py
        },
        "infer": {
            "evaluation": 3,
        },
    },

    # -----------------------------
    # NEW: Ranking
    # -----------------------------
    # JD's core mandate is "own the ranking, retrieval, and matching
    # systems." Previously zero inference rules.

    {
        "if_any": {
            "BM25",
            "Cross Encoder",
            "Bi Encoder",
            "Reranking",
        },
        "infer": {
            "ranking": 2,
        },
    },

    # -----------------------------
    # NEW: Recommendation
    # -----------------------------
    # Matching/recsys experience is explicitly called out as a strong
    # positive signal in the JD ("shipped a recommendation system at
    # a product company"). Previously zero inference rules.

    {
        "if_any": {
            "Collaborative Filtering",
            "Recommendation System",
        },
        "infer": {
            "recommendation": 3,
            "ranking": 1,
        },
    },

]