import os
import json
import asyncio
import logging
from google.generativeai.types import Tool, GenerateContentConfig, Content
from core.llm.llm_germini import get_gemini_model
from core.tools.data_extraction import data_extraction
from core.tools.data_preprocessing import preprocess_data
from core.tools.classification import classify
from core.tools.long_term_memory_retrieval import retrieve_memory
from core.tools.aggregation import aggregate_results
# from dotenv import load_dotenv

# Nhiệm vụ: Cấu hình logging để ghi lại các sự kiện và lỗi trong quá trình chạy
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Nhiệm vụ: Tải biến môi trường từ file .env (ví dụ: GOOGLE_API_KEY cho Gemini)
# load_dotenv()

async def process_traffic_line(line_number: int, csv_file: str):
    # Nhiệm vụ: Lấy mô hình Gemini từ llm_germini.py để gọi API
    model = get_gemini_model()

    # Nhiệm vụ: Đọc định nghĩa công cụ từ file config/tools.json
    try:
        with open("config/tools.json", "r") as f:
            tool_declarations = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load tools.json: {e}")
        return {"error": "Failed to load tool definitions"}

    # Nhiệm vụ: Chuyển định nghĩa JSON thành đối tượng Tool cho Gemini, hỗ trợ function calling
    tools = [Tool(function_declarations=tool_declarations)]

    # Nhiệm vụ: Chuẩn bị prompt mô tả pipeline IDS-Agent
    prompt = f"""
You are a helpful assistant for intrusion detection. Follow these steps to classify traffic from file "{csv_file}" at line {line_number}:

1. Load traffic features using the "data_extraction" tool.
2. Preprocess features using the "data_preprocessing" tool.
3. Classify using multiple classifiers with the "classifier" tool (available models: logistic_regression_ACI, decision_tree_ACI).
4. Retrieve memory using the "memory_retrieve" tool to compare results.
5. Aggregate results using the "aggregation" tool to produce a final decision with balanced sensitivity, using external knowledge if classifiers disagree.

Output must be a JSON object:
{{
  "line_number": <number>,
  "analysis": "<analysis string>",
  "predicted_label_top_1": "<string>",
  "predicted_label_top_2": "<string>",
  "predicted_label_top_3": "<string>"
}}
"""

    # Nhiệm vụ: Khởi tạo contents (STM) để lưu lịch sử tin nhắn giữa người dùng và Gemini
    contents = [Content(role="user", parts=[{"text": prompt}])]
    
    # Nhiệm vụ: Cấu hình Gemini để tự động gọi function (mode AUTO)
    config = GenerateContentConfig(tools=tools, tool_config={"function_calling_config": {"mode": "AUTO"}})

    # Nhiệm vụ: Khởi tạo biến trạng thái (STM) để lưu kết quả tạm thời
    raw_data = None
    preprocessed_data = None
    classification_results = {}
    memory = None
    reasoning_trace = []  # Lưu lịch sử suy luận (STM)

    # Nhiệm vụ: Vòng lặp xử lý function calls từ Gemini
    while True:
        # Nhiệm vụ: Gọi Gemini API để nhận phản hồi
        try:
            response = await model.generate_content(contents=contents, generation_config=config)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"error": "Failed to call Gemini API"}

        # Nhiệm vụ: Lấy candidate đầu tiên (phương án trả lời tốt nhất)
        candidate = response.candidates[0]
        
        # Nhiệm vụ: Kiểm tra nếu phản hồi là JSON kết quả cuối
        if not candidate.content.parts or not candidate.content.parts[0].function_call:
            try:
                final_result = json.loads(candidate.content.parts[0].text)
                if "line_number" in final_result:
                    logger.info(f"Final result: {final_result}")
                    return final_result
            except:
                logger.error("Invalid final response from Gemini")
                return {"error": "Invalid final response"}
            logger.error("No function call or final JSON")
            return {"error": "No function call or final JSON"}

        # Nhiệm vụ: Lấy function call từ phản hồi
        function_call = candidate.content.parts[0].function_call
        result = None

        # Nhiệm vụ: Ghi lại reasoning trace vào STM
        reasoning_trace.append(f"Called {function_call.name} with args: {function_call.args}")

        # Nhiệm vụ: Thực thi công cụ tương ứng với function call
        if function_call.name == "data_extraction":
            # Nhiệm vụ: Trích xuất dòng dữ liệu từ CSV
            result = await data_extraction(function_call.args["file_path"], function_call.args["line_number"])
            raw_data = result
            if "error" in result:
                logger.error(f"Data extraction failed: {result['error']}")
                return result
            logger.info(f"Loaded raw data: {raw_data}")
        elif function_call.name == "data_preprocessing":
            # Nhiệm vụ: Tiền xử lý dữ liệu, chuẩn hóa đặc trưng
            if raw_data is None or "error" in raw_data:
                logger.error("No valid raw data for preprocessing")
                return {"error": "No valid raw data for preprocessing"}
            result = await preprocess_data(function_call.args["raw_data"])
            preprocessed_data = result
            if "error" in result:
                logger.error(f"Data preprocessing failed: {result['error']}")
                return result
            logger.info(f"Preprocessed data: {preprocessed_data}")
        elif function_call.name == "classifier":
            # Nhiệm vụ: Phân loại dữ liệu bằng ClassificationTool
            if preprocessed_data is None or "error" in preprocessed_data:
                logger.error("No valid preprocessed data for classification")
                return {"error": "No valid preprocessed data for classification"}
            result = await classify(function_call.args["classifier_name"], function_call.args["preprocessed_features"])
            if "error" in result:
                logger.error(f"Classification failed: {result['error']}")
                return result
            classification_results[function_call.args["classifier_name"]] = result
            logger.info(f"Classifier {function_call.args['classifier_name']} result: {result}")
        elif function_call.name == "memory_retrieve":
            # Nhiệm vụ: Truy xuất LTM từ ChromaDB
            if not classification_results:
                logger.error("No classification results for memory retrieval")
                return {"error": "No classification results for memory retrieval"}
            result = await retrieve_memory(function_call.args["classifier_names"], function_call.args["classification_results"])
            memory = result
            if "error" in result:
                logger.error(f"Memory retrieval failed: {result['error']}")
                return result
            logger.info(f"Memory retrieved: {result}")
        elif function_call.name == "aggregation":
            # Nhiệm vụ: Tổng hợp kết quả, sử dụng LTM và External Documents
            if not classification_results or memory is None:
                logger.error("No classification results or memory for aggregation")
                return {"error": "No classification results or memory for aggregation"}
            result = await aggregate_results(
                function_call.args["classification_results"],
                function_call.args["memory"],
                function_call.args["line_number"],
                reasoning_trace=reasoning_trace
            )
            if "error" in result:
                logger.error(f"Aggregation failed: {result['error']}")
                return result
            logger.info(f"Aggregation result: {result}")
            return result
        else:
            logger.error(f"Unknown tool: {function_call.name}")
            return {"error": f"Unknown tool: {function_call.name}"}

        # Nhiệm vụ: Cập nhật contents (STM) với function call và kết quả
        contents.append(Content(role="model", parts=[{"function_call": function_call}]))
        contents.append(Content(role="user", parts=[{
            "function_response": {
                "name": function_call.name,
                "response": {"result": result}
            }
        }]))

# Nhiệm vụ: Chạy pipeline cho dòng dữ liệu cụ thể
if __name__ == "__main__":
    result = asyncio.run(process_traffic_line(1, "split_datasets/aci_iot_test.csv"))
    print("Final Answer:", json.dumps(result, indent=2))