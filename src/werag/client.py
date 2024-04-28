import logging
from pathlib import Path
from typing import List
from typing import Optional, Literal

from langchain.chains import LLMChain
from langchain.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from wechatpy import parse_message, create_reply

from .crud import CRUDChroma
from .db import get_chroma
from .schema import UserContent

logger = logging.getLogger(__name__)


class WeRag:
    """Core Client for werag service"""

    def __init__(self, *,
                 persist_directory: str,
                 collection_name: str = "werag",
                 embedding_function: Embeddings,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 0):
        self._crud = CRUDChroma(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._chroma = get_chroma(collection_name=collection_name, persist_directory=persist_directory,
                                  embedding_function=embedding_function)

    def as_retriever(self, *, user: str,
                     content_type: Optional[str] = None,
                     search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity",
                     **kwargs):
        return self._chroma.as_retriever(search_kwargs={
            "filter": self._crud.get_user_content_filter(user=user, content_type=content_type),
            **kwargs
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

    def response_wechat_xml(self, *, message: str,
                            llm: BaseChatModel,
                            user: str, content_type: Optional[str] = None,
                            prompt_template: Optional[str] = None,
                            ):
        """ Handle message from wechat gongzhonghao
        see: https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Receiving_standard_messages.html
        :param message: XML string from Wechat, example:
                        <xml>
                          <ToUserName><![CDATA[toUser]]></ToUserName>
                          <FromUserName><![CDATA[fromUser]]></FromUserName>
                          <CreateTime>1348831860</CreateTime>
                          <MsgType><![CDATA[text]]></MsgType>
                          <Content><![CDATA[this is a test]]></Content>
                          <MsgId>1234567890123456</MsgId>
                          <MsgDataId>xxxx</MsgDataId>
                          <Idx>xxxx</Idx>
                        </xml>
        :return:
        """

        default_prompt_template = """
        ### [INST] 
        Instruction: 回复下述问题，这里是一些数据和资料供你参考：
        
        {context}
        
        ### QUESTION:
        {question} 
        
        [/INST]
        """
        if prompt_template is None:
            prompt_template = default_prompt_template

        parsed_message = parse_message(message)
        if parsed_message.type != "text": return create_reply("我目前只能响应文字内容", parsed_message, render=True)

        # Abstraction of Prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Creating an LLM Chain

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # RAG Chain
        rag_chain = (
                {"context": self.as_retriever(user=user, content_type=content_type),
                 "question": RunnablePassthrough()}
                | llm_chain
        )
        response_text = None
        try:
            response_text = rag_chain.invoke(parsed_message.content)
            return create_reply(response_text['text'], parsed_message, render=True)
        except Exception as e:
            logger.critical(f"Failed to get response from LLM, exception: {e}, response: {response_text}")
            return create_reply("系统出错了，没有得到任何回复。请联系管理员", parsed_message, render=True)
