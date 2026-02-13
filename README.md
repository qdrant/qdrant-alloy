[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/qdrant-alloy.svg)](https://pypi.org/project/qdrant-alloy/)

# Qdrant Alloy

**Qdrant Alloy** is a high-performance, configurable hybrid search pipeline that fuses dense, sparse, and late-interaction embeddings into a single powerful retrieval engine.

Built on [Qdrant](https://github.com/qdrant/qdrant) and [FastEmbed](https://github.com/qdrant/fastembed), Alloy implements a robust **"Retrieve & Rerank"** strategy in a single unified workflow:
1.  **Dense Embeddings:** Capture semantic meaning and broad conceptual matches.
2.  **Sparse Embeddings (SPLADE/BM25):** Ensure precise keyword and lexical matching.
3.  **Late Interaction (ColBERT):** Perform fine-grained token-level reranking for superior relevance.

## Features

- 🔗 **Tri-Vector Architecture**: Seamlessly combines three vector strategies for state-of-the-art search quality.
- ⚙️ **Fully Configurable**: Swap embedding models, distance metrics, and vector parameters via simple Python config or YAML.
- 🏢 **Multi-Tenant Native**: Built-in support for partition-based multi-tenancy, perfect for SaaS applications.
- 🚀 **Production Ready**: Supports batch processing, sharding, and replication factors out of the box.

## Installation

```bash
pip install qdrant-alloy
```

Requires Python 3.11+

## Quick Start

```python
import uuid
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, KeywordIndexParams
from qdrant_alloy import AlloyConfig, Alloy

# 1. Initialize Qdrant client
# Use ":memory:" for testing or a URL for production (e.g., Qdrant Cloud)
client = QdrantClient(":memory:")

# 2. Configure embedding models
# Dense (Semantic)
text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
dense_params = VectorParams(size=text_model.dimensions, distance=Distance.COSINE)

# Sparse (Keyword)
sparse_model = SparseTextEmbedding("Qdrant/bm25")
sparse_params = SparseVectorParams(modifier=models.Modifier.IDF)

# Late Interaction (Reranking)
late_interaction_model = LateInteractionTextEmbedding("answerdotai/answerai-colbert-small-v1")
late_interaction_params = VectorParams(
    size=late_interaction_model.dimensions, 
    distance=Distance.COSINE
)

# 3. Create pipeline configuration
pipeline_config = AlloyConfig(
    text_embedding_config=(text_model, dense_params),
    sparse_embedding_config=(sparse_model, sparse_params),
    late_interaction_text_embedding_config=(late_interaction_model, late_interaction_params),
    # Optional: Multi-tenant settings
    partition_config=("tenant_id", KeywordIndexParams(minWordLength=1, maxWordLength=100)),
    multi_tenant=True,
    replication_factor=1,
    shard_number=1,
)

# 4. Initialize the pipeline
pipeline = Alloy(
    qdrant_client=client,
    collection_name="documents",
    alloy_config=pipeline_config,
)

# 5. Index documents
documents = [
    "Alloy fuses multiple search technologies into a stronger whole.",
    "Qdrant is a vector database for production-ready vector search.",
    "Late interaction models like ColBERT provide superior reranking capabilities."
]

# Metadata payloads (required for multi-tenant setups)
payloads = [
    {"tenant_id": "acme_corp", "category": "tech"},
    {"tenant_id": "acme_corp", "category": "database"},
    {"tenant_id": "acme_corp", "category": "ml"}
]

ids = [uuid.uuid4() for _ in range(len(documents))]

pipeline.insert_documents(documents=documents, payloads=payloads, document_ids=ids)

# 6. Search
# This performs dense+sparse retrieval followed by ColBERT reranking
results = pipeline.search(
    query="How does hybrid search work?", 
    top_k=3,
    partition_filter="acme_corp"
)

# 7. Process results
for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Document: {result.payload['document']}")
    print("-" * 30)
```

## How It Works

Alloy abstracts away the complexity of managing multiple embedding models and Qdrant query construction. When you run a search, Alloy performs a two-stage process:

**Prefetch (Retrieval)**: The query is embedded into both Dense and Sparse vectors. Alloy queries Qdrant to retrieve the top candidates using these vectors. This casts a wide net (semantic) while preserving specific keywords (lexical).

**Query (Rescoring)**: The retrieved candidates are immediately rescored using the Late Interaction (ColBERT) embeddings. This step examines the fine-grained interaction between query tokens and document tokens, surfacing the most relevant results to the top.

## Configuration Options

### Embedding Models

Alloy is model-agnostic and works with any model supported by fastembed:

- **Dense**: TextEmbedding (e.g., BAAI/bge-small-en-v1.5, intfloat/multilingual-e5-large)
- **Sparse**: SparseTextEmbedding (e.g., Qdrant/bm25, prithivida/Splade_PP_en_v1)
- **Late Interaction**: LateInteractionTextEmbedding (e.g., answerdotai/answerai-colbert-small-v1)

### YAML Configuration

For production deployments, define your pipeline in YAML:

```yaml
# config.yml
text_embedding:
  model_name: "BAAI/bge-small-en-v1.5"
  vector_params:
    size: 384
    distance: "Cosine"

sparse_embedding:
  model_name: "Qdrant/bm25"
  vector_params:
    modifier: "IDF"

late_interaction_text_embedding:
  model_name: "answerdotai/answerai-colbert-small-v1"
  vector_params:
    size: 128
    distance: "Cosine"

# Optional multi-tenant settings
partition_config:
  field_name: "tenant_id"
  index_params:
    type: "keyword"
    minWordLength: 1
    maxWordLength: 100

multi_tenant: true
replication_factor: 2
shard_number: 3
```

Then load it in Python:

```python
from qdrant_client import QdrantClient
from qdrant_alloy import create_alloy_from_yaml

client = QdrantClient("localhost")
config = create_alloy_from_yaml("config.yaml")
pipeline = Alloy(client, "my_collection", config)
```

## Development

To contribute to Qdrant Alloy:

```bash
# Clone the repository
git clone https://github.com/DataParthenon/qdrant-alloy.git
cd qdrant-alloy

# Install development dependencies using uv
uv venv
uv pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
