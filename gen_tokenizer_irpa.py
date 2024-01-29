from transformers import AutoTokenizer
import iree.runtime as rt
from pathlib import Path

hf_auth_token = "__Your_HF_Token__"
hf_model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(
    hf_model_name,
    use_fast=False,
    use_auth_token=hf_auth_token,
)

index = rt.ParameterIndex()
file_path = Path("tokenizer.irpa")
rt.save_archive_file(
    {
        "tokenizer": tokenizer.sp_model.serialized_model_proto(),
    },
    file_path,
)

assert(file_path.exists() and file_path.stat().st_size > 0)
