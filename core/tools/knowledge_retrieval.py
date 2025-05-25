async def memory_retrieve(classifier_names: list, results: dict):
    try:
        # Giả lập truy xuất bộ nhớ (thay bằng core/tools/long_term_memory_retrieval.py)
        return {"previous_results": {name: {"label": "Normal"} for name in classifier_names}}
    except Exception as e:
        return {"error": str(e)}