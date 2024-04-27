# import
import uuid
from typing import List
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from .schema import UserContent
from .utils import remove_none_from_dict


class CRUDChroma:

    def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def save_user_documents(self, client: Chroma, *, user: str, documents: List[Document],
                          content_type: Optional[str] = None) -> Optional[UserContent]:
        if len(documents) == 0: return None
        content = "\n".join([doc.page_content for doc in documents])
        return self.save_user_content(client=client, user=user, content=content,
                                      content_type=content_type)

    def save_user_content(self, client: Chroma, *, user: str, content: str,
                          content_type: Optional[str] = None) -> UserContent:
        # delete any existed old content with the same filter
        _filter = self.get_user_content_filter(user=user, content_type=content_type)
        old_ids = client.get(
            where=_filter
        )['ids']
        if len(old_ids) > 0: client.delete(ids=old_ids)

        # split it into chunks
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        user_content = UserContent.new_from_user_content(user=user, content=content, content_type=content_type)
        docs = text_splitter.split_documents([user_content.to_document()])

        # save
        client.add_texts(
            texts=[doc.page_content for doc in docs],
            metadatas=[remove_none_from_dict(doc.metadata) for doc in docs],
            ids=[str(uuid.uuid4()) for _ in docs]
        )
        return user_content

    def get_user_content(self, client: Chroma, *, user: str, content_type: Optional[str] = None) -> List[UserContent]:
        """Get content for user from chroma"""
        client_response = client.get(
            where=self.get_user_content_filter(user=user, content_type=content_type)
        )
        parsed = self._parse_client_response_to_user_content(client_response=client_response)
        return parsed

    def get_user_content_filter(self, *, user: str, content_type: Optional[str] = None) -> dict:
        """Get content filter base on user and additional content key

        It can be filtered by only using user, or with content type
        """
        if content_type is None: return {"user": user}
        return {
            "$and": [
                {
                    "user": {
                        "$eq": user
                    }
                },
                {
                    "content_type": {
                        "$eq": content_type
                    }
                }
            ]
        }

    def _parse_client_response_to_user_content(self, client_response: dict):
        """Parse chroma client response to user content object"""
        return [UserContent(
            id=client_response['ids'][index],
            user=client_response['metadatas'][index]['user'],
            content_type=client_response['metadatas'][index].get("content_type"),
            page_content=client_response['documents'][index]
        ) for index, _ in enumerate(client_response['ids'])]

    def search_documents(self, client: Chroma, *, query: str, limit: int = 4, user: str,
                         content_type: Optional[str] = None) -> \
            List[Document]:
        return client.similarity_search(query, k=limit,
                                        filter=self.get_user_content_filter(user=user, content_type=content_type))
