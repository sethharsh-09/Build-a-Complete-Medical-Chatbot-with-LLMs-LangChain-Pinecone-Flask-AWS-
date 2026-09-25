import os
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from pinecone_text.sparse import BM25Encoder

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

INDEX_NAME = os.environ.get(
    "PINECONE_INDEX_NAME",
    "medical-chatbot-hybrid"
)

NAMESPACE = os.environ.get(
    "PINECONE_NAMESPACE",
    "medical"
)


# =========================================================
# CONSTANTS
# =========================================================

DENSE_DIMENSION = 384

DATA_PATH = "data/"

ARTIFACT_DIR = Path("artifacts")

BM25_PATH = ARTIFACT_DIR / "bm25_params.json"


# =========================================================
# CHECK API KEY
# =========================================================

if not PINECONE_API_KEY:

    raise ValueError(
        "PINECONE_API_KEY is missing from .env"
    )


# =========================================================
# CREATE ARTIFACT DIRECTORY
# =========================================================

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. LOAD PDF
# =========================================================

print("Loading PDF files...")

extracted_data = load_pdf_file(
    data=DATA_PATH
)

print(
    f"Loaded {len(extracted_data)} PDF pages."
)


# =========================================================
# 2. FILTER METADATA
# =========================================================

filter_data = filter_to_minimal_docs(
    extracted_data
)


# =========================================================
# 3. SPLIT INTO CHUNKS
# =========================================================

text_chunks = text_split(
    filter_data
)

print(
    f"Created {len(text_chunks)} text chunks."
)


# =========================================================
# 4. EXTRACT TEXT
# =========================================================

corpus = [
    chunk.page_content
    for chunk in text_chunks
]


# =========================================================
# 5. DENSE EMBEDDING MODEL
# =========================================================

print("Loading dense embedding model...")

embeddings = download_hugging_face_embeddings()

print("Dense model loaded.")


# =========================================================
# 6. CREATE DENSE VECTORS
# =========================================================

print("Creating dense vectors...")

dense_vectors = embeddings.embed_documents(
    corpus
)

print(
    f"Created {len(dense_vectors)} dense vectors."
)


# =========================================================
# 7. CREATE SPARSE BM25 MODEL
# =========================================================

print("Training BM25 sparse encoder...")

bm25 = BM25Encoder(
    lower_case=True,
    remove_punctuation=False,
    remove_stopwords=False,
    stem=False
)

bm25.fit(corpus)

print("BM25 model fitted.")


# =========================================================
# 8. SAVE BM25 PARAMETERS
# =========================================================

bm25.dump(
    str(BM25_PATH)
)

print(
    f"BM25 parameters saved to {BM25_PATH}"
)


# =========================================================
# 9. CREATE SPARSE VECTORS
# =========================================================

print("Creating sparse vectors...")

sparse_vectors = bm25.encode_documents(
    corpus
)

print(
    f"Created {len(sparse_vectors)} sparse vectors."
)


# =========================================================
# 10. CONNECT TO PINECONE
# =========================================================

print("Connecting to Pinecone...")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# =========================================================
# 11. CREATE HYBRID INDEX
# =========================================================

if not pc.has_index(INDEX_NAME):

    print(
        f"Creating Pinecone index: {INDEX_NAME}"
    )

    pc.create_index(
        name=INDEX_NAME,

        # Dense vector dimension
        dimension=DENSE_DIMENSION,

        # Hybrid search uses dot product
        metric="dotproduct",

        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Pinecone index created.")

else:

    print(
        f"Pinecone index '{INDEX_NAME}' already exists."
    )


# =========================================================
# 12. CONNECT TO INDEX
# =========================================================

index = pc.Index(
    INDEX_NAME
)


# =========================================================
# 13. DELETE OLD DOCUMENT VECTORS
# =========================================================
#
# This is important for the assignment.
#
# If we run store_index.py again,
# old chunks of the same PDF are removed first.
#
# Therefore duplicate records are not created.
# =========================================================

sources = sorted(
    {
        chunk.metadata.get(
            "source",
            "unknown"
        )
        for chunk in text_chunks
    }
)


for source in sources:

    print(
        f"Removing old vectors for: {source}"
    )

    index.delete(
        filter={
            "source": {
                "$eq": source
            }
        },
        namespace=NAMESPACE
    )


# =========================================================
# 14. PREPARE RECORDS
# =========================================================

records = []

for i, chunk in enumerate(text_chunks):

    source = chunk.metadata.get(
        "source",
        "unknown"
    )

    text = chunk.page_content

    # -----------------------------------------------------
    # Deterministic document ID
    # -----------------------------------------------------

    source_hash = hashlib.sha1(
        source.encode("utf-8")
    ).hexdigest()[:16]

    vector_id = (
        f"{source_hash}-chunk-{i}"
    )

    # -----------------------------------------------------
    # Dense vector
    # -----------------------------------------------------

    dense_vector = dense_vectors[i]

    # -----------------------------------------------------
    # Sparse vector
    # -----------------------------------------------------

    sparse_vector = sparse_vectors[i]

    # -----------------------------------------------------
    # Pinecone record
    # -----------------------------------------------------

    record = {
        "id": vector_id,

        "values": dense_vector,

        "sparse_values": {
            "indices": sparse_vector["indices"],
            "values": sparse_vector["values"]
        },

        "metadata": {
            "text": text,
            "source": source,
            "chunk_index": i
        }
    }

    records.append(record)


# =========================================================
# 15. UPSERT IN BATCHES
# =========================================================

BATCH_SIZE = 100

print(
    f"Upserting {len(records)} hybrid vectors..."
)


for start in range(
    0,
    len(records),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(records)
    )

    batch = records[start:end]

    index.upsert(
        vectors=batch,
        namespace=NAMESPACE
    )

    print(
        f"Uploaded {start + 1} - {end}"
    )


# =========================================================
# 16. FINISHED
# =========================================================

print()
print("=" * 60)
print("HYBRID INDEXING COMPLETED")
print("=" * 60)

print(
    f"Index     : {INDEX_NAME}"
)

print(
    f"Namespace : {NAMESPACE}"
)

print(
    f"Chunks    : {len(records)}"
)

print(
    "Dense     : all-MiniLM-L6-v2"
)

print(
    "Sparse    : BM25"
)

print(
    "Status    : SUCCESS"
)

print("=" * 60)
