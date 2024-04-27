from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from werag.crud import CRUDChroma
from .utils import prune_chroma, get_client
from .llm import get_llm
crud = CRUDChroma(chunk_size=1000)
client = get_client()


def test_save_user_documents():
    prune_chroma(client._chroma)

    user = "user1"
    content = "content1"
    content_type = "content_type1"
    client.save_content(user=user, content=content, content_type=content_type)
    doc = client._chroma.get(
        where={
            "user": user
        }
    )
    assert len(doc['documents']) > 0

    assert doc['documents'][0] == content
    assert doc['metadatas'][0]['user'] == user
    assert doc['metadatas'][0]["content_type"] == content_type


def test_save_documents():
    prune_chroma(client._chroma)

    user = "user1"
    content = "content1_for_document"
    content_type = "content_type1"
    client.save_documents(user=user, documents=[Document(page_content=content)], content_type=content_type)
    doc = client._chroma.get(
        where={
            "user": user
        }
    )
    assert len(doc['documents']) > 0

    assert doc['documents'][0] == content
    assert doc['metadatas'][0]['user'] == user
    assert doc['metadatas'][0]["content_type"] == content_type


def test_import_files():
    prune_chroma(client._chroma)
    with open("assets/lorem.txt", mode="r") as f:
        content = f.read()

    user = "user_lorem"

    content_type = "lorem_content_type"
    client.import_files(user=user, filepaths=["./assets/lorem.txt"], content_type=content_type)
    doc = client._chroma.get(
        where={
            "user": user
        }
    )
    assert len(doc['documents']) > 0

    assert doc['documents'][0] in content
    assert doc['metadatas'][0]['user'] == user
    assert doc['metadatas'][0]["content_type"] == content_type


def test_client_data_persist():
    prune_chroma(client._chroma)
    with open("assets/lorem.txt", mode="r") as f:
        content = f.read()

    # create some data
    user = "user_lorem"
    content_type = "lorem_content_type"
    client.import_files(user=user, filepaths=["./assets/lorem.txt"], content_type=content_type)
    doc = client._chroma.get(
        where={
            "user": user
        }
    )
    assert len(doc['documents']) > 0

    # create another client
    new_client = get_client()
    new_doc = new_client._chroma.get(
        where={
            "user": user
        }
    )
    assert len(new_doc['documents']) > 0
    assert len(new_doc['documents']) == len(doc['documents'])
    for c in new_doc['documents']:
        assert c in content


def test_save_as_url():
    prune_chroma(client._chroma)
    urls = [
        "https://www.lipsum.com/"
    ]
    # create some data
    user = "user_lorem"
    content_type = "lorem_content_type"
    client.save_urls(user=user, urls=urls, content_type=content_type)
    doc = client._chroma.get(
        where={
            "user": user
        }
    )
    assert len(doc['documents']) > 0
    found_key_phrase = False
    key_phrases = ["Why do we use it?", "dummy text of the printing and typesetting"]
    for c in doc['documents']:
        for kp in key_phrases:
            if kp in c:
                found_key_phrase = True
                break

    assert found_key_phrase


def test_client_as_retriever():
    # create data
    prune_chroma(client._chroma)
    user = "user"
    content_type = "personal"
    client.import_files(user=user, filepaths=["./assets/personal_info.txt"], content_type=content_type)
    # test llm
    prompt_template = """
    ### [INST] 
    Instruction: Answer the question based on your 
    human biology and anatomy knowledge. Here is context to help:

    {context}

    ### QUESTION:
    {question} 

    [/INST]
    """

    # Abstraction of Prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Creating an LLM Chain

    llm_chain = LLMChain(llm=get_llm(), prompt=prompt)

    # RAG Chain
    rag_chain = (
            {"context": client.as_retriever(user=user, content_type=content_type), "question": RunnablePassthrough()}
            | llm_chain
    )

    # query
    query_1 = "What is my name?"

    response = rag_chain.invoke(query_1)
    assert "zhangwei" in response['text'].lower()

