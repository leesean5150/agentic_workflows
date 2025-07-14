def extract_response_llama(response: str) -> str:
    """
    Efficiently extract the last assistant response from a LLaMA-formatted string.
    Assumes messages use LLaMA chat template tags.
    """
    start_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    end_marker = "<|eot_id|>"

    start_idx = response.rfind(start_marker)
    
    if start_idx == -1:
        return None

    start_idx += len(start_marker)

    return response[start_idx:].strip()
