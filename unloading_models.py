import gc
import torch


def clear_cuda_and_gc_mem():
    """
    Force Python garbage collection and release cached CUDA memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def validate_allocated_memory():
    """
    Print current CUDA allocated and reserved memory, if CUDA is available.
    """
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated()}")
        print(f"Memory reserved: {torch.cuda.memory_reserved()}")
    else:
        print("[validate_allocated_memory] torch.cuda is not available")


def unload_pipeline(pipe):
    """
    Remove all pipeline-held references to a model and free CUDA memory.
    """
    print("[unload_pipeline] Check if all references to the model have been deleted.")

    # Move model to CPU first, if present
    if hasattr(pipe, "model") and pipe.model is not None:
        try:
            pipe.model.to("cpu")
        except Exception:
            pass

    # Clear pipeline references
    if hasattr(pipe, "model"):
        pipe.model = None
    if hasattr(pipe, "tokenizer"):
        pipe.tokenizer = None
    if hasattr(pipe, "feature_extractor"):
        pipe.feature_extractor = None
    if hasattr(pipe, "image_processor"):
        pipe.image_processor = None
    if hasattr(pipe, "processor"):
        pipe.processor = None

    del pipe

    clear_cuda_and_gc_mem()


def unload_model_and_tokenizer(model, tokenizer=None):
    """
    Move a standalone model to CPU, delete references, and free CUDA memory.
    """
    try:
        model.to("cpu")
    except Exception:
        pass

    del model
    if tokenizer is not None:
        del tokenizer

    clear_cuda_and_gc_mem()
