import os

from flask import (
    Flask,
    render_template,
    request
)

from dotenv import load_dotenv

from pinecone import Pinecone

from pinecone_text.sparse import BM25Encoder

from langchain_openai import ChatOpenAI

from langchain.schema import Document

from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from src.helper import (
    download_hugging_face_embeddings
)

from src.prompt import system_prompt


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


PINECONE_API_KEY = os.environ.get(
    "PINECONE_API_KEY"
)

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY"
)

INDEX_NAME = os.environ.get(
    "PINECONE_INDEX_NAME",
    "medical-chatbot-hybrid"
)

NAMESPACE = os.environ.get(
    "PINECONE_NAMESPACE",
    "medical"
)

HYBRID_ALPHA = float(
    os.environ.get(
        "HYBRID_ALPHA",
        "0.65"
    )
)

TOP_K = int(
    os.environ.get(
        "TOP_K",
        "4"
    )
)


# =========================================================
# VALIDATE ENVIRONMENT
# =========================================================

if not PINECONE_API_KEY:

    raise ValueError(
        "PINECONE_API_KEY is missing."
    )


if not OPENAI_API_KEY:

    raise ValueError(
        "OPENAI_API_KEY is missing."
    )


# =========================================================
# PINECONE CONNECTION
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


# =========================================================
# CHECK INDEX
# =========================================================

if not pc.has_index(INDEX_NAME):

    raise RuntimeError(
        f"Pinecone index '{INDEX_NAME}' "
        "does not exist. "
        "Run 'python store_index.py' first."
    )


index = pc.Index(
    INDEX_NAME
)


# =========================================================
# DENSE EMBEDDING MODEL
# =========================================================

print(
    "Loading dense embedding model..."
)

embeddings = (
    download_hugging_face_embeddings()
)

print(
    "Dense embedding model loaded."
)


# =========================================================
# LOAD BM25 SPARSE MODEL
# =========================================================

BM25_PATH = (
    "artifacts/bm25_params.json"
)


if not os.path.exists(BM25_PATH):

    raise FileNotFoundError(
        "BM25 model not found. "
        "Run 'python store_index.py' first."
    )


bm25 = BM25Encoder(
    lower_case=True,
    remove_punctuation=False,
    remove_stopwords=False,
    stem=False
)

bm25.load(
    BM25_PATH
)


print(
    "BM25 sparse encoder loaded."
)


# =========================================================
# HYBRID VECTOR SCALING
# =========================================================

def hybrid_scale(
    dense_vector,
    sparse_vector,
    alpha=0.65
):
    """
    Combine dense and sparse vectors.

    alpha = 1.0 -> only dense
    alpha = 0.0 -> only sparse
    """

    if not 0 <= alpha <= 1:

        raise ValueError(
            "Alpha must be between 0 and 1."
        )


    # -----------------------------------------------------
    # Dense contribution
    # -----------------------------------------------------

    scaled_dense = [
        value * alpha
        for value in dense_vector
    ]


    # -----------------------------------------------------
    # Sparse contribution
    # -----------------------------------------------------

    scaled_sparse = {

        "indices": sparse_vector[
            "indices"
        ],

        "values": [
            value * (1 - alpha)
            for value in sparse_vector[
                "values"
            ]
        ]
    }


    return (
        scaled_dense,
        scaled_sparse
    )


# =========================================================
# HYBRID SEARCH FUNCTION
# =========================================================

def hybrid_search(
    question,
    top_k=TOP_K,
    alpha=HYBRID_ALPHA
):
    """
    Perform dense + sparse hybrid search.
    """

    # -----------------------------------------------------
    # 1. Generate dense query vector
    # -----------------------------------------------------

    dense_vector = (
        embeddings.embed_query(
            question
        )
    )


    # -----------------------------------------------------
    # 2. Generate sparse query vector
    # -----------------------------------------------------

    sparse_vector = (
        bm25.encode_queries(
            question
        )
    )


    # -----------------------------------------------------
    # 3. Scale vectors
    # -----------------------------------------------------

    hybrid_dense, hybrid_sparse = (
        hybrid_scale(
            dense_vector,
            sparse_vector,
            alpha
        )
    )


    # -----------------------------------------------------
    # 4. Query Pinecone
    # -----------------------------------------------------

    result = index.query(

        vector=hybrid_dense,

        sparse_vector=hybrid_sparse,

        top_k=top_k,

        include_metadata=True,

        namespace=NAMESPACE
    )


    return result


# =========================================================
# LLM
# =========================================================

chatModel = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",
            system_prompt
        ),

        (
            "human",
            "{input}"
        )
    ]
)


# =========================================================
# DOCUMENT QA CHAIN
# =========================================================

question_answer_chain = (
    create_stuff_documents_chain(
        chatModel,
        prompt
    )
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():

    return render_template(
        "chat.html"
    )


# =========================================================
# CHAT API
# =========================================================

@app.route(
    "/get",
    methods=["GET", "POST"]
)
def chat():

    try:

        # -------------------------------------------------
        # GET USER QUESTION
        # -------------------------------------------------

        msg = request.form.get(
            "msg",
            ""
        ).strip()


        if not msg:

            return "Please enter a question."


        print(
            f"User Question: {msg}"
        )


        # -------------------------------------------------
        # HYBRID SEARCH
        # -------------------------------------------------

        search_result = hybrid_search(
            question=msg,
            top_k=TOP_K,
            alpha=HYBRID_ALPHA
        )


        # -------------------------------------------------
        # CONVERT PINECONE RESULTS
        # TO LANGCHAIN DOCUMENTS
        # -------------------------------------------------

        documents = []


        for match in search_result.matches:

            metadata = match.metadata or {}

            text = metadata.get(
                "text",
                ""
            )


            if not text:

                continue


            documents.append(

                Document(

                    page_content=text,

                    metadata={
                        "source": metadata.get(
                            "source",
                            ""
                        ),

                        "score": float(
                            match.score
                        )
                    }
                )
            )


        # -------------------------------------------------
        # NO DOCUMENT FOUND
        # -------------------------------------------------

        if not documents:

            return (
                "I don't know the answer "
                "from the available medical "
                "documents."
            )


        # -------------------------------------------------
        # SEND CONTEXT + QUESTION TO GPT-4o
        # -------------------------------------------------

        response = (
            question_answer_chain.invoke(
                {
                    "input": msg,
                    "context": documents
                }
            )
        )


        print(
            "Response:",
            response
        )


        return str(response)


    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

        return (
            "Sorry, an error occurred "
            "while processing your question."
        )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
