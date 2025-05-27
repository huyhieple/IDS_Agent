import pandas as pd
import logging
from typing import Union, Dict

logger = logging.getLogger(__name__)

async def data_extraction(file_path: str, identifier: Union[int, str], by: str = "line_number") -> Dict:
    """
    Extracts a traffic record from a CSV file based on either line number or flow ID.

    Args:
        file_path (str): Path to the CSV file.
        identifier (Union[int, str]):  The line number (1-based index) or the flow ID 
                                      of the record to extract.
        by (str, optional):  Specify "line_number" or "flow_id" to indicate how to 
                            extract the record. Defaults to "line_number".

    Returns:
        Dict: A dictionary containing the traffic record, or an error message.
    """

    try:
        df = pd.read_csv(file_path)

        if by == "line_number":
            if not isinstance(identifier, int):
                raise ValueError("Identifier must be an integer when by='line_number'")
            if identifier < 1 or identifier > len(df):
                raise ValueError(f"Line number {identifier} out of range for file {file_path}")
            row = df.iloc[identifier - 1].to_dict()  # 0-based indexing
        elif by == "flow_id":
            if not isinstance(identifier, str):
                raise ValueError("Identifier must be a string when by='flow_id'")
            
            # Find the row where 'Flow ID' column equals the identifier
            row = df.loc[df['Flow ID'] == identifier].to_dict(orient='records')
            if not row:
                raise ValueError(f"Flow ID '{identifier}' not found in file {file_path}")
            row = row[0]  # Extract the dictionary, .loc returns a list
        else:
            raise ValueError(f"Invalid value for 'by': {by}. Must be 'line_number' or 'flow_id'")

        logger.info(f"Extracted data from {file_path} by {by} '{identifier}': {row}")
        return row

    except FileNotFoundError:
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return {"error": error_msg}
    except ValueError as ve:
        error_msg = str(ve)
        logger.error(error_msg)
        return {"error": error_msg}
    except KeyError as ke:
        error_msg = f"Column '{ke}' not found in CSV file."
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Data extraction failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    
# Đầu vào
#data_extraction("traffic\_data.csv", 230, "line_number"): Trong trường hợp này, 
# identifier là số nguyên 230, hàm sẽ trích xuất dữ liệu từ dòng 230 của file "traffic_data.csv".

#data_extraction("traffic\_data.csv", "XYZ123", "flow_id"): Trong trường hợp này, 
# identifier là chuỗi "XYZ123", hàm sẽ trích xuất dữ liệu của bản ghi có Flow ID là "XYZ123".

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