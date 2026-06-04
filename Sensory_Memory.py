import math
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import AppConfig, get_config


class SensoryMemory:
    def __init__(
        self,
        qdrant_client: Any,
        collection_name: Optional[str] = None,
        config: Optional[AppConfig] = None,
        load_model: bool = True,
    ):
        self.config = config or get_config()
        self.qdrant = qdrant_client
        self.collection_name = collection_name or self.config.sensory_collection_name
        self.model_name = self.config.sensory_model_name
        self.model_source = self._resolve_model_source()
        self.processor = None
        self.model = None
        self.torch = None
        self.device = "cpu"
        
        if load_model:
            self._load_model()

    def _resolve_model_source(self) -> str:
        local_path = self.config.sensory_model_local_path
        if local_path:
            expanded = Path(os.path.expanduser(local_path)).resolve()
            if expanded.exists():
                return str(expanded)
        return self.model_name

    def _from_pretrained_kwargs(self) -> Dict[str, Any]:
        kwargs = {"local_files_only": self.config.sensory_model_local_files_only}
        if self.config.hf_token:
            kwargs["token"] = self.config.hf_token
        return kwargs

    def _load_model(self):
        import torch
        from transformers import ChineseCLIPProcessor, ChineseCLIPModel

        print(f"正在加载开源中文 CLIP 模型: {self.model_source}")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = self._from_pretrained_kwargs()
        try:
            self.processor = ChineseCLIPProcessor.from_pretrained(self.model_source, **kwargs)
            self.model = ChineseCLIPModel.from_pretrained(self.model_source, **kwargs)
        except Exception as exc:
            if kwargs.get("local_files_only"):
                raise RuntimeError(
                    "中文 CLIP 模型本地加载失败。请确认 SENSORY_MODEL_LOCAL_PATH 指向完整模型目录，"
                    "或确认 HuggingFace 缓存中已有该模型；如允许联网下载，可设置 "
                    "SENSORY_MODEL_LOCAL_FILES_ONLY=false。"
                ) from exc
            raise
        self.model.to(self.device)
        self.model.eval()

    def _ensure_model_loaded(self):
        if self.processor is None or self.model is None:
            raise RuntimeError("SensoryMemory 未加载中文 CLIP 模型，请使用 load_model=True 初始化。")

    def get_image_embedding(self, image_path: str) -> List[float]:
        """获取图像向量"""
        self._ensure_model_loaded()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # 归一化处理
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.squeeze().tolist()

    def get_text_embedding(self, text: str) -> List[float]:
        """获取文本向量 (映射到与图片相同的空间)"""
        self._ensure_model_loaded()
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze().tolist()

    def retrieve(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """跨模态混合检索，使用你的打分公式"""
        search_results = self._vector_search(
            query_vector=query_vector,
            limit=top_k * 3,
        )

        final_results = []
        for hit in search_results:
            vector_sim = hit.score
            timestamp = hit.payload.get("timestamp", time.time())
            importance = hit.payload.get("importance", 5.0) # 默认中等重要性
            
            # 计算时间衰减 (指数衰减)
            time_diff = (time.time() - timestamp) / 3600
            recency = math.exp(-0.01 * time_diff)
            
            # 你的核心感知召回公式
            final_score = (vector_sim * 0.8 + recency * 0.2) * (0.8 + importance * 0.4)
            
            final_results.append({
                "id": hit.id,
                "payload": hit.payload,
                "score": final_score
            })
            
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

    def _vector_search(self, query_vector: List[float], limit: int):
        if hasattr(self.qdrant, "search"):
            return self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
        if hasattr(self.qdrant, "query_points"):
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
            )
            return getattr(response, "points", response)
        raise AttributeError(
            "当前 QdrantClient 既没有 search，也没有 query_points，"
            "请检查 qdrant-client 版本或传入兼容的客户端。"
        )
