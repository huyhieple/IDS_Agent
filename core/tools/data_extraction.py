import pandas as pd
import logging

logger = logging.getLogger(__name__)

async def data_extraction(file_path: str, line_number: int):
    """
    Load traffic features from a CSV file for a specific line.
    
    Args:
        file_path: Path to the CSV file (e.g., 'split_datasets/aci_iot_test.csv').
        line_number: Line number to extract (1-based indexing).
    
    Returns:
        Dict containing the features of the specified line, or error if failed.
    """
    try:
        # Đọc file CSV
        df = pd.read_csv(file_path)
        
        # Kiểm tra số dòng hợp lệ
        if line_number < 1 or line_number > len(df):
            raise ValueError(f"Line number {line_number} out of range for file {file_path}")
        
        # Trích xuất dòng (chuyển sang 0-based indexing)
        row = df.iloc[line_number - 1].to_dict()
        
        logger.info(f"Extracted data from {file_path} at line {line_number}: {row}")
        return row
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        return {"error": str(e)}
    

# -----> RETURN row (1 hàng chứa dữ liệu)
# ví dụ 
#{
#  "Flow ID": 2,
#  "Src IP": "10.0.0.3",
#  "Dst IP": "10.0.0.4",
#  "Protocol": "UDP",
#  "Flow Duration": 78910,
#  "Label": "Attack"
#}