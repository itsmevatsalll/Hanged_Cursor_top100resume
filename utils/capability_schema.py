"""
Capability Knowledge Base

Each capability contains:
1. Explicit skills
2. Contextual phrases
3. Related technologies

The detector will search ALL resume sections and
compute evidence scores.

FIX: production_ml and evaluation previously had very narrow vocabularies
(production_ml: 2 skills + 8 phrases, evaluation: 0 skills + 10 phrases).
The JD names these as the #1 "absolutely need" categories, so under-
detecting them due to narrow vocabulary directly undermines ranking
quality. Expanded both sections below with common real-world terms for
ML deployment and ranking evaluation. All other capabilities unchanged
from the original.
"""

CAPABILITIES = {

    # --------------------------------------------------------
    # Retrieval Systems
    # --------------------------------------------------------

    "retrieval": {
        "skills": {
            "RAG",
            "FAISS",
            "Milvus",
            "Pinecone",
            "Qdrant",
            "Weaviate",
            "Sentence Transformers",
            "Embeddings",
            "BM25",
            "Learning to Rank",
            "Information Retrieval",
        },

        "phrases": {
            "semantic search",
            "vector search",
            "hybrid retrieval",
            "dense retrieval",
            "sparse retrieval",
            "cross encoder",
            "bi encoder",
            "reranking",
            "nearest neighbour",
            "approximate nearest neighbour",
            "document search",
            "knowledge retrieval",
            "context retrieval",
            "embedding search",
            "similarity search",
        },
    },

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    "llm": {
        "skills": {
            "LLMs",
            "LangChain",
            "LlamaIndex",
            "Prompt Engineering",
            "Fine-tuning LLMs",
            "LoRA",
            "QLoRA",
        },

        "phrases": {
            "large language model",
            "foundation model",
            "instruction tuning",
            "parameter efficient fine tuning",
            "prompt engineering",
            "prompt optimization",
            "retrieval augmented generation",
            "rag pipeline",
            "agentic workflow",
            "llama",
            "mistral",
            "gemma",
            "openai api",
        },
    },

    # --------------------------------------------------------
    # Backend
    # --------------------------------------------------------

    "backend": {
        "skills": {
            "FastAPI",
            "Flask",
            "Django",
            "Spring Boot",
            "REST APIs",
            "Node.js",
            "GraphQL",
        },

        "phrases": {
            "backend service",
            "microservice",
            "distributed service",
            "rest api",
            "grpc",
            "api gateway",
            "authentication",
            "authorization",
            "jwt",
            "load balancer",
        },
    },

    # --------------------------------------------------------
    # Cloud
    # --------------------------------------------------------

    "cloud": {
        "skills": {
            "AWS",
            "Azure",
            "GCP",
        },

        "phrases": {
            "cloud deployment",
            "cloud infrastructure",
            "serverless",
            "ec2",
            "lambda",
            "docker",
            "kubernetes",
            "terraform",
            "autoscaling",
        },
    },

    # --------------------------------------------------------
    # Data Engineering
    # --------------------------------------------------------

    "data_engineering": {
        "skills": {
            "Spark",
            "Kafka",
            "Airflow",
            "Snowflake",
            "BigQuery",
        },

        "phrases": {
            "data pipeline",
            "etl",
            "stream processing",
            "batch processing",
            "real time pipeline",
            "streaming pipeline",
            "feature pipeline",
        },
    },

    # --------------------------------------------------------
    # Ranking Systems
    # --------------------------------------------------------

    "ranking": {
        "skills": {
            "Learning to Rank",
            "XGBoost",
            "LambdaMART",
            "LightGBM",
        },

        "phrases": {
            "ranking system",
            "candidate ranking",
            "search ranking",
            "re ranking",
            "reranking",
            "ranking model",
            "ndcg",
            "mrr",
            "map",
            "precision at k",
            "recall at k",
        },
    },

    # --------------------------------------------------------
    # Recommendation Systems
    # --------------------------------------------------------

    "recommendation": {
        "skills": {
            "Recommendation System",
            "Collaborative Filtering",
        },

        "phrases": {
            "recommendation engine",
            "recommendation system",
            "personalization",
            "candidate matching",
            "matching engine",
            "search and recommendation",
        },
    },

    # --------------------------------------------------------
    # Production ML  (EXPANDED)
    # --------------------------------------------------------

    "production_ml": {
        "skills": {
            "MLflow",
            "Kubeflow",
            "TorchServe",
            "TensorFlow Serving",
            "BentoML",
            "Seldon",
            "SageMaker",
            "Vertex AI",
            "Triton Inference Server",
        },

        "phrases": {
            "production ml",
            "production deployment",
            "deployed to production",
            "production inference",
            "online serving",
            "offline evaluation",
            "model serving",
            "model monitoring",
            "model registry",
            "model versioning",
            "canary deployment",
            "shadow deployment",
            "mlops",
            "ci cd for ml",
            "feature store",
            "embedding drift",
            "index refresh",
            "retrieval quality regression",
            # FIX: removed "real users" and "production system" — these
            # are generic enough to false-positive on plain backend/infra
            # descriptions with zero ML content (e.g. "scaled the API to
            # serve real users"). Also removed "scaled to millions of
            # users" / "shipped to production" for the same reason —
            # neither implies ML specifically. Kept the rest, which are
            # all ML-deployment-specific terms unlikely to appear outside
            # an ML context.
        },
    },

    # --------------------------------------------------------
    # Evaluation  (EXPANDED)
    # --------------------------------------------------------

    "evaluation": {
        "skills": {
            "A/B Testing",
            "Offline Evaluation",
        },

        "phrases": {
            "ndcg",
            "mrr",
            "map",
            "ab testing",
            "a/b testing",
            "offline benchmark",
            "online evaluation",
            "evaluation framework",
            "retrieval quality",
            "ranking quality",
            "precision at k",
            "recall at k",
            "offline to online correlation",
            "ground truth labels",
            "human evaluation",
            "eval harness",
            "relevance judgments",
            "click through rate",
            "conversion rate",
        },
    },

    # --------------------------------------------------------
    # Distributed Systems
    # --------------------------------------------------------

    "distributed": {
        "skills": {
            "Spark",
            "Kafka",
        },

        "phrases": {
            "distributed system",
            "distributed computing",
            "large scale",
            "horizontal scaling",
            "fault tolerant",
            "high throughput",
        },
    },

    # --------------------------------------------------------
    # Open Source
    # --------------------------------------------------------

    "opensource": {
        "skills": {
            "GitHub",
        },

        "phrases": {
            "open source",
            "github",
            "pull request",
            "contributor",
            "maintainer",
        },
    },
}