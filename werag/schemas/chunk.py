from datetime import datetime
from typing import List, Optional

from keble_chains import AiPrompt, Vector
from pydantic import BaseModel

from .base import HasPrompts
from .raw import RawType


class AigenConv:
    class Base(HasPrompts):
        raw: Optional[str] = None  # raw record's point id
        type: RawType  # AigenConvType
        time: Optional[datetime] = None
        metadata: dict

        def to_qdrant_payload(self):
            return AigenConv.QdrantPayload(content=self.content,
                                           key=self.key,
                                           **self.model_dump())

    class QdrantPayload(Base):
        content: str
        key: str

    class QdrantVector(BaseModel):
        content: Vector

    class QdrantVectorPayload(BaseModel):
        content: str