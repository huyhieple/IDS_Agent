import chromadb
import logging
import json
from google.generativeai import GenerativeModel
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Nhiệm vụ: Khởi tạo logger và tải biến môi trường
logger = logging.getLogger(__name__)
load_dotenv()

class AggregationTool:
    def __init__(self, collection_name="iot_attacks", model_name="all-MiniLM-L6-v2"):
        # Nhiệm vụ: Khởi tạo ChromaDB client và collection
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Nhiệm vụ: Khởi tạo encoder và LLM
        self.encoder = SentenceTransformer(model_name)
        self.llm = GenerativeModel("gemini-2.0-flash")  # Hoặc model khác

    def summarize_documents(self, documents: list) -> str:
        """
        Tóm tắt danh sách tài liệu bằng LLM.
        """
        try:
            # Nhiệm vụ: Tạo prompt để tóm tắt
            prompt = f"Summarize the following documents in 50 words or less:\n\n{'\n'.join(documents)}"
            
            # Nhiệm vụ: Gọi Gemini để tóm tắt
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to summarize documents: {str(e)}")
            return "Summary unavailable."

    async def aggregate_results(self, classification_results: dict, memory: dict, line_number: int, reasoning_trace: list) -> dict:
        """
        Tổng hợp kết quả phân loại, sử dụng LTM và tài liệu bên ngoài.
        """
        try:
            # Nhiệm vụ: Lấy tất cả nhãn từ classification_results
            labels = []
            for classifier, result in classification_results.items():
                labels.append(result["predicted_label_top_1"])
                if result["predicted_label_top_2"]:
                    labels.append(result["predicted_label_top_2"])
                if result["predicted_label_top_3"]:
                    labels.append(result["predicted_label_top_3"])
            labels = list(set(labels))

            # Nhiệm vụ: Tạo truy vấn cho vector database
            query = " ".join(labels) + " attack characteristics"
            query_embedding = self.encoder.encode(query).tolist()

            # Nhiệm vụ: Truy xuất top-3 tài liệu
            results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
            documents = results["documents"][0] if results["documents"] else []

            # Nhiệm vụ: Tóm tắt tài liệu
            summary = self.summarize_documents(documents) if documents else "No external knowledge retrieved."

            # Nhiệm vụ: Tổng hợp nhãn
            top_label = max(labels, key=labels.count, default="Unknown")
            label_counts = {label: labels.count(label) for label in labels}
            sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
            
            # Nhiệm vụ: Tạo phân tích
            analysis = (
                f"Analyzed line {line_number} with {', '.join(classification_results.keys())}. "
                f"Balanced sensitivity applied. Memory: {json.dumps(memory)}. "
                f"Classification results: {json.dumps(classification_results)}. "
                f"External knowledge: {summary}. Reasoning trace: {json.dumps(reasoning_trace)}."
            )

            # Nhiệm vụ: Trả về JSON kết quả
            return {
                "line_number": line_number,
                "analysis": analysis,
                "predicted_label_top_1": sorted_labels[0] if len(sorted_labels) > 0 else "Unknown",
                "predicted_label_top_2": sorted_labels[1] if len(sorted_labels) > 1 else "Unknown",
                "predicted_label_top_3": sorted_labels[2] if len(sorted_labels) > 2 else "Unknown"
            }
        
        except Exception as e:
            logger.error(f"Aggregation failed: {str(e)}")
            return {"error": str(e)}