import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_config
from rag.bm25_index import BM25ChunkIndex


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the local BM25 index from Qdrant RAG chunks.")
    parser.add_argument("--collection", default=None, help="Qdrant collection name. Defaults to RAG_COLLECTION_NAME.")
    parser.add_argument("--output", default=None, help="BM25 index output path. Defaults to RAG_BM25_INDEX_PATH.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    config = get_config()
    collection = args.collection or config.rag_collection_name
    output = args.output or config.rag_bm25_index_path
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout=config.qdrant_timeout,
    )
    index = BM25ChunkIndex.rebuild_from_qdrant(
        qdrant_client=qdrant,
        collection_name=collection,
        embed_model_name=config.embed_model_name,
        batch_size=args.batch_size,
    )
    index.save(output)
    print(f"BM25 index rebuilt: documents={len(index.documents)}, collection={collection}, output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
