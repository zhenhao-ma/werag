from werag.crud import CRUDChroma
from .utils import get_chroma, prune_chroma

crud = CRUDChroma(chunk_size=1000)
chroma = get_chroma()


def test_save_user_content_short_content():
    prune_chroma(chroma)
    user = "user1"
    content = "content1"
    crud.save_user_content(client=chroma, user=user, content=content)

    doc = chroma.get(
        where={
            "user": user
        }
    )

    assert doc['documents'][0] == content
    assert doc['metadatas'][0]['user'] == user
    assert "content_type" not in doc['metadatas'][0]


def test_save_user_content_long_content():
    prune_chroma(chroma)
    with open("assets/lorem.txt", mode="r") as f:
        content = f.read()

    user = "user1"

    crud.save_user_content(client=chroma, user=user, content=content)

    doc = chroma.get(
        where={
            "user": user
        }
    )
    assert len(doc['ids']) > 1

    for index, _ in enumerate(doc['ids']):
        assert doc['documents'][index] in content
        assert doc['metadatas'][index]['user'] == user
        assert "content_type" not in doc['metadatas'][index]


def test_save_user_content_with_content_type():
    # prune_chroma(chroma)
    user = "user1"
    content = "content1"
    content_type = "type1"
    crud.save_user_content(client=chroma, user=user, content=content, content_type=content_type)

    doc = chroma.get(
        where={
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
    )
    assert doc['documents'][0] == content
    assert doc['metadatas'][0]['user'] == user
    assert doc['metadatas'][0]["content_type"] == content_type


def test_get_user_content():
    prune_chroma(chroma)
    with open("assets/lorem.txt", mode="r") as f:
        content = f.read()

    user = "user1"
    content_type = "lorem"
    prevent_content_type = "prevent_type"
    prevent_content = "prevent_content"
    crud.save_user_content(client=chroma, user=user, content=content, content_type=content_type)
    crud.save_user_content(client=chroma, user=user, content=prevent_content, content_type=prevent_content_type)

    contents = crud.get_user_content(client=chroma, user=user, content_type=content_type)

    for c in contents:
        assert c.page_content in content
        assert prevent_content not in c.page_content
        assert c.user == user
        assert c.content_type == content_type

    all_contents = crud.get_user_content(client=chroma, user=user)
    found_prevent = False
    found_non_prevent = False
    for c in all_contents:
        if c.content_type == prevent_content_type:
            found_prevent = True
        else:
            found_non_prevent = True
    assert found_non_prevent is True
    assert found_prevent is True

def test_search_user_content():
    prune_chroma(chroma)
    content = "lorem2"
    content_type = "lorem_type"
    prevent_content_type = "lorem1"
    prevent_content = "lorem_type2"
    user = "me"
    crud.save_user_content(client=chroma, user=user, content=content, content_type=content_type)
    crud.save_user_content(client=chroma, user=user, content=prevent_content, content_type=prevent_content_type)

    # search prevent, should return content (not prevent content)
    docs = crud.search_documents(chroma, query=prevent_content_type, limit=2, user=user, content_type=content_type)
    # should only return 1 doc
    assert len(docs) == 1
    assert docs[0].page_content == content
    assert docs[0].metadata['user'] == user
    assert docs[0].metadata['content_type'] == content_type