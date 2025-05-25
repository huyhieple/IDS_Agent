import pickle
import numpy as np
import os
from typing import Dict, List, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ClassificationTool:
    """
    Công cụ để tải và sử dụng các mô hình machine learning đã được huấn luyện.
    """
    def __init__(self, model_dir: str = 'D:/UIT 2025-2026/Hocmay/Project/IDS_Agent/models/ACI_IoT_23/'):
        """
        Khởi tạo ClassificationTool.

        Args:
            model_dir: Đường dẫn đến thư mục chứa các file .pkl của mô hình.
        """
        self.model_dir = model_dir
        self.loaded_models: Dict[str, object] = {}  # Cache các model đã load

    def load_model(self, model_name: str) -> object:
        """
        Tải một mô hình từ file .pkl.

        Args:
            model_name: Tên của mô hình (không bao gồm phần mở rộng .pkl).

        Returns:
            Mô hình đã được tải.

        Raises:
            ValueError: Nếu không tìm thấy file mô hình.
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_path = os.path.join(self.model_dir, f'{model_name}.pkl')

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                self.loaded_models[model_name] = model
                return model
        except FileNotFoundError:
            raise ValueError(f"Không tìm thấy mô hình: {model_name}")

    def predict(self, model_name: str, features: Union[List[float], List[List[float]], np.ndarray]) -> Dict[str, Union[List, List[List], None]]:
        """
        Thực hiện dự đoán bằng mô hình đã tải.

        Args:
            model_name: Tên của mô hình cần sử dụng.
            features: Dữ liệu đầu vào cho mô hình. Có thể là một sample hoặc nhiều samples.

        Returns:
            Một dictionary chứa kết quả dự đoán, tuân theo định dạng JSON trong bài báo:
            {
                'predicted_label_top_1': str,
                'predicted_label_top_2': str,
                'predicted_label_top_3': str,
                'probabilities': list (nếu có)
            }
            Trả về None nếu có lỗi xảy ra.

        Raises:
            ValueError: Nếu có lỗi liên quan đến đầu vào.
            RuntimeError: Nếu có lỗi trong quá trình dự đoán.
        """
        model = self.load_model(model_name)
        try:
            features = np.array(features)
            if features.ndim == 1:
                features = features.reshape(1, -1)

            predictions = model.predict(features)
            top_3_labels = self._get_top_k_labels(model, features, k=3)

            result: Dict[str, Union[List, List[List], None]] = {
                'predicted_label_top_1': top_3_labels[0],
                'predicted_label_top_2': top_3_labels[1],
                'predicted_label_top_3': top_3_labels[2],
                'probabilities': None
            }

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)
                result['probabilities'] = probabilities.tolist()

            return result

        except ValueError as ve:
            raise ValueError(f"Lỗi đầu vào: {ve}")
        except RuntimeError as e:
            raise RuntimeError(f"Lỗi khi dự đoán: {e}")

    def _get_top_k_labels(self, model, features, k=3):
        """
        Lấy ra top-k nhãn dự đoán và xác suất tương ứng (nếu có).

        Args:
            model: Mô hình đã được tải.
            features: Dữ liệu đầu vào.
            k: Số lượng nhãn top-k cần lấy.

        Returns:
            Danh sách các nhãn top-k.
        """
        predictions = model.predict(features)
        top_k_labels = predictions.tolist()

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            top_k_indices = np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]
            all_labels = model.classes_
            top_k_labels = []
            for i in range(len(features)):
                top_labels_for_sample = [all_labels[idx] for idx in top_k_indices[i]]
                top_k_labels.append(top_labels_for_sample)
        else:
            top_k_labels = predictions.tolist()

        return top_k_labels[0] if len(features) == 1 else top_k_labels

# Hàm classify để tích hợp với main.py
async def classify(classifier_name: str, preprocessed_features: dict) -> Dict[str, Union[str, List, None]]:
    """
    Phân loại đặc trưng đã tiền xử lý bằng ClassificationTool.

    Args:
        classifier_name: Tên mô hình (e.g., 'logistic_regression_ACI', 'decision_tree_ACI').
        preprocessed_features: Dictionary chứa đặc trưng đã tiền xử lý.

    Returns:
        Dictionary chứa kết quả dự đoán hoặc lỗi.
    """
    try:
        # Khởi tạo ClassificationTool
        tool = ClassificationTool()
        
        # Loại bỏ trường Label nếu có
        features = {k: v for k, v in preprocessed_features.items() if k != "Label"}
        
        # Sắp xếp đặc trưng theo thứ tự
        feature_names = [
            'Src Port', 'Dst Port', 'Protocol', 'Flow IAT Mean', 'Flow IAT Max',
            'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
            'SYN Flag Count', 'RST Flag Count', 'Down/Up Ratio',
            'Subflow Fwd Packets', 'FWD Init Win Bytes', 'Fwd Seg Size Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Flow Bytes/s',
            'Connection Type'
        ]
        feature_values = [features.get(f, 0) for f in feature_names]
        
        # Gọi predict từ ClassificationTool
        result = tool.predict(classifier_name, feature_values)
        
        if result is None:
            raise ValueError(f"Prediction failed for {classifier_name}")
        
        logger.info(f"Classification result for {classifier_name}: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        return {"error": str(e)}