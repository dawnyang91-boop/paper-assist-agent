from typing import Optional


class QdrantVectorDimensionError(ValueError):
    pass


def get_collection_vector_size(qdrant, collection_name: str) -> Optional[int]:
    """Return the single-vector collection size when it can be inspected."""
    try:
        info = qdrant.get_collection(collection_name=collection_name)
    except Exception:
        return None

    vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
    if vectors is None:
        return None

    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size

    if isinstance(vectors, dict):
        for value in vectors.values():
            named_size = getattr(value, "size", None)
            if isinstance(named_size, int):
                return named_size
            if isinstance(value, dict) and isinstance(value.get("size"), int):
                return value["size"]
    return None


def ensure_collection_vector_size(qdrant, collection_name: str, vector_size: int, distance=None) -> None:
    """Create a collection or fail early when an existing collection has another dimension."""
    try:
        exists = qdrant.collection_exists(collection_name=collection_name)
    except Exception:
        return

    if exists:
        existing_size = get_collection_vector_size(qdrant, collection_name)
        if existing_size is not None and existing_size != vector_size:
            raise QdrantVectorDimensionError(
                f"Qdrant collection '{collection_name}' 向量维度不一致："
                f"当前 collection={existing_size}，当前配置/模型输出={vector_size}。"
                "请重建 collection，或修改 RAG_COLLECTION_NAME 使用新的 collection 后重新 ingest。"
            )
        return

    try:
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        return

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance or Distance.COSINE),
    )
