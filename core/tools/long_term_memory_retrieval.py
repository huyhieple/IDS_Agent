import chromadb
import logging
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import json

# Nhiệm vụ: Khởi tạo logger
logger = logging.getLogger(__name__)

class LTMRetriever:
    def __init__(self, collection_name="ltm_iot", model_name="all-MiniLM-L6-v2"):
        # Nhiệm vụ: Khởi tạo ChromaDB client và collection
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Nhiệm vụ: Khởi tạo encoder để tạo embedding
        self.encoder = SentenceTransformer(model_name)
        
        # Nhiệm vụ: Đặt trọng số cho thời gian và cosine similarity
        self.lambda1 = 0.5  # Trọng số thời gian
        self.lambda2 = 0.5  # Trọng số cosine similarity

    def add_ltm_entry(self, entry: dict):
        """
        Thêm một mục LTM vào ChromaDB.
        """
        try:
            # Nhiệm vụ: Mã hóa observations thành chuỗi JSON
            observations_str = json.dumps(entry["O"])
            
            # Nhiệm vụ: Tạo embedding cho observations
            embedding = self.encoder.encode(observations_str).tolist()
            
            # Nhiệm vụ: Lưu mục LTM vào ChromaDB
            self.collection.add(
                documents=[observations_str],
                embeddings=[embedding],
                metadatas=[{
                    "timestamp": entry["t"],
                    "features": json.dumps(entry["x"]),
                    "reasoning_trace": json.dumps(entry["R"]),
                    "actions": json.dumps(entry["A"]),
                    "final_label": entry["ŷ"]
                }],
                ids=[f"ltm_{entry['t']}"]
            )
            logger.info(f"Added LTM entry with timestamp {entry['t']}")
        except Exception as e:
            logger.error(f"Failed to add LTM entry: {str(e)}")

    async def retrieve_memory(self, classifier_names: list, classification_results: dict) -> dict:
        """
        Truy xuất top-5 mục LTM liên quan.
        """
        try:
            # Nhiệm vụ: Mã hóa classification_results thành chuỗi JSON
            observations_str = json.dumps(classification_results)
            embedding = self.encoder.encode(observations_str).tolist()
            
            # Nhiệm vụ: Lấy thời gian hiện tại
            current_time = datetime.now().isoformat()
            current_timestamp = datetime.fromisoformat(current_time).timestamp()

            # Nhiệm vụ: Lấy tất cả mục LTM
            all_entries = self.collection.get(include=["metadatas", "embeddings"])
            
            if not all_entries["ids"]:
                return {"previous_results": []}

            # Nhiệm vụ: Tính điểm cho từng mục
            scores = []
            for idx, metadata in enumerate(all_entries["metadatas"]):
                # Tính độ gần thời gian
                entry_time = datetime.fromisoformat(metadata["timestamp"]).timestamp()
                time_diff = abs(current_timestamp - entry_time)
                max_diff = max(abs(datetime.fromisoformat(m["timestamp"]).timestamp() - current_timestamp) for m in all_entries["metadatas"])
                recency = 1 - (time_diff / max_diff) if max_diff > 0 else 1
                
                # Tính cosine similarity
                entry_embedding = all_entries["embeddings"][idx]
                cos_sim = np.dot(embedding, entry_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(entry_embedding))
                
                # Tổng hợp điểm
                score = self.lambda1 * recency + self.lambda2 * cos_sim
                scores.append((score, metadata))

            # Nhiệm vụ: Lấy top-5
            scores.sort(reverse=True)
            top_k = scores[:5]
            
            previous_results = [
                {
                    "timestamp": entry["timestamp"],
                    "features": json.loads(entry["features"]),
                    "final_label": entry["final_label"]
                }
                for _, entry in top_k
            ]
            
            return {"previous_results": previous_results}
        
        except Exception as e:
            logger.error(f"Memory retrieval failed: {str(e)}")
            return {"error": str(e)}