import pandas as pd
import json
import logging
import asyncio
from datetime import datetime
from core.tools.long_term_memory_retrieval import LTMRetriever
from core.tools.aggregation import AggregationTool
from core.tools.data_preprocessing import preprocess_data
from core.tools.classification import classify

# Nhiệm vụ: Khởi tạo logger để ghi log
logger = logging.getLogger(__name__)

async def initialize_ltm(validation_file: str):
    """
    Khởi tạo Long-term Memory (LTM) từ tập xác thực.
    
    Args:
        validation_file: Đường dẫn đến file CSV xác thực (e.g., 'train/aci_iot_train.csv').
    """
    try:
        # Nhiệm vụ: Đọc file xác thực
        df = pd.read_csv(validation_file)
        
        # Nhiệm vụ: Khởi tạo LTMRetriever để lưu trữ LTM vào ChromaDB
        ltm_retriever = LTMRetriever()

        # Nhiệm vụ: Duyệt qua từng dòng trong file xác thực
        for idx, row in df.iterrows():
            # Nhiệm vụ: Chuyển dòng thành dictionary (raw_data)
            raw_data = row.to_dict()
            
            # Nhiệm vụ: Tiền xử lý dữ liệu để tạo preprocessed_data
            preprocessed_data = await preprocess_data(raw_data)
            if "error" in preprocessed_data:
                logger.warning(f"Preprocessing failed for row {idx}: {preprocessed_data['error']}")
                continue
            
            # Nhiệm vụ: Tạo classification_results bằng ClassificationTool
            classification_results = {
                "logistic_regression_ACI": await classify("logistic_regression_ACI", preprocessed_data),
                "decision_tree_ACI": await classify("decision_tree_ACI", preprocessed_data)
            }
            if any("error" in result for result in classification_results.values()):
                logger.warning(f"Classification failed for row {idx}")
                continue
            
            # Nhiệm vụ: Tạo reasoning_trace và actions (STM)
            reasoning_trace = ["data_extraction", "data_preprocessing", "classifier", "memory_retrieve", "aggregation"]
            actions = reasoning_trace
            
            # Nhiệm vụ: Tạo observations (quan sát từ pipeline)
            observations = [raw_data, preprocessed_data, classification_results]

            # Nhiệm vụ: Tạo mục LTM (ϕ)
            entry = {
                "t": datetime.now().isoformat(),  # Thời gian hiện tại
                "x": [preprocessed_data.get(f, 0.0) for f in [
                    'Src Port', 'Dst Port', 'Protocol', 'Flow IAT Mean', 'Flow IAT Max',
                    'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
                    'SYN Flag Count', 'RST Flag Count', 'Down/Up Ratio',
                    'Subflow Fwd Packets', 'FWD Init Win Bytes', 'Fwd Seg Size Min',
                    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Flow Bytes/s',
                    'Connection Type'
                ]],  # Đặc trưng thực từ preprocessed_data
                "R": reasoning_trace,  # Lịch sử suy luận
                "A": actions,  # Danh sách hành động
                "O": observations,  # Danh sách quan sát
                "ŷ": classification_results["logistic_regression_ACI"]["predicted_label_top_1"]  # Nhãn dự đoán
            }
            
            # Nhiệm vụ: Lưu mục LTM vào ChromaDB
            ltm_retriever.add_ltm_entry(entry)
        
        logger.info(f"Initialized LTM with {len(df)} entries")
    
    except Exception as e:
        logger.error(f"Failed to initialize LTM: {str(e)}")

def initialize_external_documents():
    """
    Khởi tạo vector database với tài liệu mẫu.
    """
    try:
        # Nhiệm vụ: Khởi tạo AggregationTool
        aggregation_tool = AggregationTool()
        
        # Nhiệm vụ: Tạo tài liệu mẫu
        sample_documents = [
            "DoS attacks flood network with traffic, causing service disruption.",
            "Benign traffic follows normal patterns, low risk.",
            "SYN Flood exploits TCP handshake, overwhelming servers.",
            "Port Scan probes open ports to identify vulnerabilities."
        ]
        
        # Nhiệm vụ: Tạo embedding và lưu vào ChromaDB
        embeddings = aggregation_tool.encoder.encode(sample_documents).tolist()
        aggregation_tool.collection.add(
            documents=sample_documents,
            embeddings=embeddings,
            ids=[f"doc_{i}" for i in range(len(sample_documents))]
        )
        logger.info("Initialized external documents")
    
    except Exception as e:
        logger.error(f"Failed to initialize external documents: {str(e)}")

# Nhiệm vụ: Chạy khởi tạo LTM và External Documents
if __name__ == "__main__":
    # Nhiệm vụ: Chạy hàm initialize_ltm bất đồng bộ
    asyncio.run(initialize_ltm("train/aci_iot_train.csv"))
    initialize_external_documents()