from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def load_local_hf_model(
    model_path: str,
    device: str = None,
    max_new_tokens: int = 200,
    temperature: float = 0.1,
    top_p: float = 0.9,
    stop_token: str = None
) -> HuggingFacePipeline:
    """
    Load a Hugging Face model locally and wrap it for LangChain, with support for:
    - Mac M-series via MPS
    - NVIDIA GPUs via CUDA
    - CPU fallback

    Args:
        model_path (str): Local path or model ID (cached).
        device (str, optional): 'auto' | 'cuda' | 'mps' | 'cpu'. Auto-detect if None.
        max_new_tokens (int): Max tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.

    Returns:
        HuggingFacePipeline: Wrapped pipeline for use in LangChain.
    """

    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    terminators = list(
        filter(
            lambda x: x is not None and x != tokenizer.unk_token_id,
            [tokenizer.eos_token_id] +
            ([tokenizer.convert_tokens_to_ids(stop_token)] if stop_token else [])
        )
    )

    if device in ["cuda", "mps"]:
        model = model.to(device)
    else:
        model = model.to("cpu")

    pipeline_device = 0 if device == "cuda" else -1

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=pipeline_device,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=terminators
    )

    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=gen_pipeline), model_id=model_path)
