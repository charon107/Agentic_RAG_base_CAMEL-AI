import os


def check_database_exists() -> bool:
    """Check whether the Qdrant vector database collection exists.

    Returns:
        bool: True if the collection directory exists and is non-empty.
    """
    db_path = os.path.join("src", "qdrant_data", "collection", "rag_collection")
    return os.path.exists(db_path) and len(os.listdir(db_path)) > 0

