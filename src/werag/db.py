from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_chroma(*, collection_name: str, persist_directory: str,
               embedding_function: Embeddings) -> Chroma:
    # create the chroma client
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )
