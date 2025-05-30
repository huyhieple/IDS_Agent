import chromadb
import logging
import json
from google.generativeai import GenerativeModel
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import numpy as np
from typing import Dict, List, Union

# Nhiệm vụ: Khởi tạo logger và tải biến môi trường
logger = logging.getLogger(__name__)
load_dotenv()

# Định nghĩa trọng số cho các bộ phân loại (có thể điều chỉnh dựa trên hiệu suất)
CLASSIFIER_WEIGHTS = {
    "logistic_regression_ACI": 0.4,
    "decision_tree_ACI": 0.3,
    "multi_layer_perceptrons_ACI": 0.3,
    "k_nearest_neighbors_ACI": 0.25,
    "random_forest_ACI": 0.35,
    "svc_ACI": 0.3
}

class AggregationTool:
    def __init__(self, collection_name="iot_attacks", model_name="all-MiniLM-L6-v2", db_path="D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/ChromaDB/"):
        # Nhiệm vụ: Khởi tạo ChromaDB client và collection
        try:
            os.makedirs(db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logger.info(f"Connected to ChromaDB collection '{collection_name}' at '{db_path}'.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.collection = None

        # Nhiệm vụ: Khởi tạo encoder và LLM
        try:
            self.encoder = SentenceTransformer(model_name)
            self.llm = GenerativeModel("gemini-2.0-flash")
            logger.info(f"Initialized SentenceTransformer '{model_name}' and Gemini LLM.")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer or LLM: {e}")
            self.encoder = None
            self.llm = None

    def summarize_documents(self, documents: List[str]) -> str:
        """
        Tóm tắt danh sách tài liệu bằng LLM.
        """
        if not self.llm:
            logger.error("LLM not initialized.")
            return "Summary unavailable."
        try:
            joined_documents = '\n'.join(documents)
            prompt = f"Summarize the following documents in 50 words or less:\n\n{joined_documents}"
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Failed to summarize documents: {e}")
            return "Summary unavailable."

    async def aggregate_results(self, classification_results: Dict[str, Dict], memory: Dict, line_number: int, reasoning_trace: List[str]) -> Dict[str, Union[str, List, None]]:
        """
        Tổng hợp kết quả phân loại, sử dụng LTM và tri thức bên ngoài.
        """
        try:
            if not classification_results:
                logger.error("No classification results provided.")
                return {"error": "No classification results provided."}

            # Nhiệm vụ: Lấy tất cả nhãn từ classification_results
            labels = []
            probabilities = {}
            for classifier, result in classification_results.items():
                if "error" in result:
                    logger.warning(f"Classifier '{classifier}' returned error: {result['error']}")
                    continue
                labels.append(result["predicted_label_top_1"])
                if result["predicted_label_top_2"]:
                    labels.append(result["predicted_label_top_2"])
                if result["predicted_label_top_3"]:
                    labels.append(result["predicted_label_top_3"])
                probabilities[classifier] = result.get("probabilities", [1.0, 0.0, 0.0])

            if not labels:
                logger.error("No valid labels extracted from classification results.")
                return {"error": "No valid labels extracted."}

            # Nhiệm vụ: Đếm tần suất và kiểm tra tie
            label_counts = {label: labels.count(label) for label in labels}
            max_count = max(label_counts.values())
            tied_labels = [label for label, count in label_counts.items() if count == max_count]

            # Nhiệm vụ: Phá vỡ thế cân bằng nếu có tie
            top_label = None
            if len(tied_labels) > 1:
                logger.info(f"Tied labels detected: {tied_labels}. Using LTM and external knowledge for tie-breaking.")
                # Sử dụng LTM
                from core.tools.long_term_memory_retrieval import LTMRetriever
                ltm_retriever = LTMRetriever()
                memory_result = await ltm_retriever.retrieve_memory(
                    list(classification_results.keys()), classification_results
                )
                ltm_labels = [entry["final_label"] for entry in memory_result.get("previous_results", [])]
                ltm_counts = {label: ltm_labels.count(label) for label in tied_labels if label in ltm_labels}
                if ltm_counts and max(ltm_counts.values()) > 0:
                    top_label = max(ltm_counts, key=ltm_counts.get)
                    logger.info(f"Selected '{top_label}' based on LTM counts: {ltm_counts}")
                else:
                    # Sử dụng tri thức bên ngoài từ KnowledgeRetrievalTool
                    from core.tools.knowledge_retrieval import knowledge_retrieval_tool_function
                    query = " ".join(tied_labels) + " attack characteristics"
                    knowledge = await knowledge_retrieval_tool_function(query)
                    external_summary = knowledge.get("retrieved_knowledge", "").lower()
                    for label in tied_labels:
                        if label.lower() in external_summary:
                            top_label = label
                            logger.info(f"Selected '{top_label}' based on external knowledge.")
                            break
                    if not top_label:
                        top_label = min(tied_labels)  # Fallback: alphabet
                        logger.warning(f"No tie-breaker found. Selected '{top_label}' by alphabetical order.")
            else:
                top_label = max(labels, key=labels.count, default="Unknown")
                logger.info(f"Selected '{top_label}' based on majority voting.")

            # Nhiệm vụ: Tạo truy vấn cho ChromaDB
            query = " ".join(set(labels)) + " attack characteristics"
            query_embedding = self.encoder.encode(query).tolist() if self.encoder else None
            documents = []
            if self.collection and query_embedding:
                try:
                    results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
                    documents = results["documents"][0] if results["documents"] else []
                except Exception as e:
                    logger.error(f"Error querying ChromaDB: {e}")

            # Nhiệm vụ: Tóm tắt tài liệu
            summary = self.summarize_documents(documents) if documents else "No external knowledge retrieved."

            # Nhiệm vụ: Truy xuất tri thức bên ngoài
            from core.tools.knowledge_retrieval import knowledge_retrieval_tool_function
            try:
                external_knowledge = await knowledge_retrieval_tool_function(query)
                external_summary = external_knowledge.get("retrieved_knowledge", "No external knowledge retrieved.")
            except Exception as e:
                logger.error(f"Error retrieving external knowledge: {e}")
                external_summary = "External knowledge retrieval failed."

            # Nhiệm vụ: Tổng hợp nhãn với xác suất
            weighted_scores = {label: 0.0 for label in label_counts}
            for classifier, result in classification_results.items():
                if "error" in result:
                    continue
                for i, label_key in enumerate(["predicted_label_top_1", "predicted_label_top_2", "predicted_label_top_3"]):
                    label = result.get(label_key)
                    if label and label in weighted_scores:
                        prob = result.get("probabilities", [1.0, 0.0, 0.0])[i] if i < len(result.get("probabilities", [])) else 0.0
                        weighted_scores[label] += prob * CLASSIFIER_WEIGHTS.get(classifier, 1.0)

            sorted_labels = sorted(weighted_scores, key=weighted_scores.get, reverse=True)

            # Nhiệm vụ: Tạo phân tích
            analysis = (
                f"Analyzed line {line_number} with {', '.join(classification_results.keys())}. "
                f"Balanced sensitivity applied. Memory: {json.dumps(memory, ensure_ascii=False)}. "
                f"Classification results: {json.dumps(classification_results, ensure_ascii=False)}. "
                f"ChromaDB knowledge: {summary}. External knowledge: {external_summary}. "
                f"Reasoning trace: {json.dumps(reasoning_trace, ensure_ascii=False)}."
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
            logger.error(f"Aggregation failed: {e}", exc_info=True)
            return {"error": str(e)}