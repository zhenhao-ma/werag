from uuid import uuid4

from langchain_core.documents import Document
from pydantic import BaseModel
from typing import List, Optional

class UserContent(BaseModel):
    """Model for user content that stored in chroma database"""
    page_content: str
    content_type: Optional[str]
    user: str
    id: str

    def to_document(self) -> Document:
        """convert object to langchain Document object (pydantic v1)"""
        return Document(
            page_content=self.page_content,
            metadata={
                "user": self.user,
                "content_type": self.content_type
            }
        )

    @classmethod
    def new_from_user_content(cls, *, user: str, content: str, content_type: Optional[str] = None) -> "UserContent":
        """Build new object from new user content"""
        return UserContent(
            user=user,
            page_content=content,
            content_type=content_type,
            id=str(uuid4())
        )



