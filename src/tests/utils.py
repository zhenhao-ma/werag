from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.embeddings import Embeddings

from werag import WeRag


def get_embedding_function() -> Embeddings:
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


collection_name = "werag__pytest"


def get_chroma() -> Chroma:
    # create the chroma client
    return Chroma(
        collection_name=collection_name,
        persist_directory="./chroma_persist",
        embedding_function=get_embedding_function()
    )


def get_client() -> WeRag:
    return WeRag(
        persist_directory="./chroma_persist",
        collection_name=collection_name,
        chunk_size=1000,
        chunk_overlap=0
    )


def prune_chroma(chroma: Chroma):
    ids = chroma.get(where={})['ids']
    if len(ids) > 0:
        chroma.delete(ids=ids)
