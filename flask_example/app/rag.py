import os

from werag import WeRag


def get_werag(app):
    folder = os.path.join(app.instance_path, "werag")
    return WeRag(
        persist_directory=folder,
        collection_name="werag"
    )
