import json

async def aggregate_results(classification_results: dict, memory: dict, line_number: int):
    """
    Aggregate classification results from multiple classifiers to produce a final decision.
    
    Args:
        classification_results: Dict containing classifier names and their results (label, confidence).
        memory: Dict containing previous classification results from memory retrieval.
        line_number: Line number of the traffic data.
    
    Returns:
        Dict in JSON format with line_number, analysis, and top-3 predicted labels.
    """
    try:
        labels = [r["label"] for r in classification_results.values() if "label" in r]
        analysis = (
            f"Analyzed line {line_number} with {', '.join(classification_results.keys())}. "
            f"Balanced sensitivity applied. Memory: {json.dumps(memory.get('previous_results', {}))}. "
            f"Classification results: {json.dumps(classification_results)}."
        )
        return {
            "line_number": line_number,
            "analysis": analysis,
            "predicted_label_top_1": labels[0] if labels else "Unknown",
            "predicted_label_top_2": labels[1] if len(labels) > 1 else "Unknown",
            "predicted_label_top_3": labels[2] if len(labels) > 2 else "Unknown"
        }
    except Exception as e:
        return {"error": f"Aggregation failed: {str(e)}"}