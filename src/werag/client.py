from pathlib import Path
from typing import List
from typing import Optional, Literal

from langchain.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .crud import CRUDChroma
from .db import get_chroma
from .schema import UserContent


class WeRag:
    """Core Client for werag service"""

    def __init__(self, *,
                 persist_directory: str,
                 collection_name: str = "werag",
                 embedding_function: Optional[Embeddings] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 0):
        self._crud = CRUDChroma(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if embedding_function is None:
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self._chroma = get_chroma(collection_name=collection_name, persist_directory=persist_directory,
                                  embedding_function=embedding_function)

    def as_retriever(self, *, user: str,
                     content_type: Optional[str] = None,
                     search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity",
                     limit: int = 4,
                     score_threshold: float = 0.8,
                     fetch_k: int = 20,
                     lambda_mult: float = 0.5):
        return self._chroma.as_retriever(search_kwargs={
            # "k": limit,
            # "score_threshold": score_threshold,
            "filter": self._crud.get_user_content_filter(user=user, content_type=content_type),
            # "fetch_k": fetch_k,
            # "lambda_mult": lambda_mult
        }, search_type=search_type)

    def save_content(self, *, user: str, content: str,
                     content_type: Optional[str] = None) -> UserContent:
        """Save a content base on user id"""
        return self._crud.save_user_content(client=self._chroma, user=user, content_type=content_type, content=content)

    def save_documents(self, *, user: str, documents: List[Document],
                       content_type: Optional[str] = None) -> Optional[UserContent]:
        """Save a docs base on user id"""

        return self._crud.save_user_documents(client=self._chroma, user=user, documents=documents,
                                              content_type=content_type)

    def save_urls(self, *, urls: List[str], user: str,
                  max_depth: int = 1,
                  content_type: Optional[str] = None, **kwargs) -> Optional[UserContent]:
        docs = []
        for url in urls:
            loader = RecursiveUrlLoader(url, max_depth=max_depth, **kwargs)
            docs += loader.load()

        # Converts HTML to plain text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        return self.save_documents(user=user, documents=docs_transformed, content_type=content_type)

    def import_files(self, *, filepaths: List[str | Path], user: str,
                     content_type: Optional[str] = None) -> Optional[UserContent]:
        """Import content from files"""
        docs = []
        for filepath in filepaths:
            loader = TextLoader(filepath)
            docs += loader.load()
        return self.save_documents(user=user, content_type=content_type, documents=docs)
