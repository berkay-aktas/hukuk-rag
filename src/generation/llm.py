"""Qwen2.5-7B loader and generation utilities.

Supports two modes via a single :func:`load_llm` entrypoint:

- **Base**: vanilla Qwen2.5-7B-Instruct in 4-bit NF4.
- **QLoRA**: same base + a trained adapter merged in via PEFT.

Critical inference-time fixes are baked into :func:`generate_answer` and
documented as code comments rather than memory notes, because the original
gradient_checkpointing + use_cache conflict cost 5 hours of compute and a
2-hour eval run before being caught. Keep them in code where they survive
any session loss.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class LoadedLLM:
    """A loaded model + tokenizer pair, ready for generation."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    is_qlora: bool
    base_model_id: str
    adapter_path: str | None = None

    def unload(self) -> None:
        """Free VRAM and CPU memory held by this model."""
        del self.model
        del self.tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def load_llm(
    base_model: str = DEFAULT_LLM,
    adapter_path: str | Path | None = None,
    *,
    load_in_4bit: bool = True,
    device_map: str = "auto",
) -> LoadedLLM:
    """Load Qwen2.5 base model with optional QLoRA adapter.

    Args:
        base_model: HuggingFace model id (default Qwen2.5-7B-Instruct).
        adapter_path: Optional path to a PEFT adapter directory (e.g. Drive
            path to ``models/qwen-qlora-v2/adapter``). If provided, loads as QLoRA.
        load_in_4bit: NF4 double quantization. Required on T4/L4 (16 GB VRAM).
            Set False only if you have ≥24 GB VRAM and want full bf16.
        device_map: HuggingFace device mapping. ``"auto"`` lets accelerate decide.

    Returns:
        A :class:`LoadedLLM` with model, tokenizer, and a flag indicating QLoRA.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading base model: %s (4-bit=%s)", base_model, load_in_4bit)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
    )

    is_qlora = False
    if adapter_path is not None:
        from peft import PeftModel

        logger.info("Loading QLoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        is_qlora = True

    # CRITICAL: At inference time, gradient_checkpointing must be OFF and
    # use_cache must be ON. If gradient_checkpointing was enabled during
    # training (it must be, to fit Qwen-7B on T4), it persists into the
    # loaded model and breaks generation — output becomes garbage like
    # "KastenDemocratsence kasten复工复%@\",死去". This cost 5h+ of compute
    # in a prior session. Do not remove these two lines.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    return LoadedLLM(
        model=model,
        tokenizer=tokenizer,
        is_qlora=is_qlora,
        base_model_id=base_model,
        adapter_path=str(adapter_path) if adapter_path else None,
    )


def generate_answer(
    llm: LoadedLLM,
    system_prompt: str,
    user_message: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    do_sample: bool | None = None,
) -> str:
    """Generate a single answer from the LLM using the chat template.

    Args:
        llm: A :class:`LoadedLLM` from :func:`load_llm`.
        system_prompt: System role content.
        user_message: User role content (typically question + context block).
        max_new_tokens: Generation cap. 256 fits the +49% baseline reproduction;
            raise to 512 for verbose comparisons.
        temperature: Sampling temperature. 0.1 = near-greedy, deterministic.
        top_p: Nucleus sampling.
        repetition_penalty: 1.2 suppresses the QLoRA degenerate-repetition
            artifact (token distribution drift between base and adapter).
            Do not lower below 1.1 for QLoRA inference.
        do_sample: Force-override sampling. Defaults to ``True`` if temperature > 0.

    Returns:
        The generated assistant text, stripped.
    """
    import torch

    if do_sample is None:
        do_sample = temperature > 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.model.device)

    with torch.inference_mode():
        outputs = llm.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=llm.tokenizer.eos_token_id,
        )

    # Strip the prompt prefix — only return the newly generated tokens.
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return llm.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def smoke_test(llm: LoadedLLM) -> str:
    """One-prompt sanity check for QLoRA inference health.

    Run this immediately after loading a QLoRA adapter, BEFORE any benchmark.
    If output contains garbage characters (CJK, repetition, broken Turkish),
    the gradient_checkpointing fix isn't applied or the adapter is corrupt.
    """
    from src.generation.prompts import DEFAULT_SYSTEM

    test_q = "Kasten adam öldürme suçunun cezası nedir?"
    answer = generate_answer(
        llm,
        system_prompt=DEFAULT_SYSTEM,
        user_message=f"Soru: {test_q}\n\nBağlam: (yok, test sorusu)",
        max_new_tokens=128,
    )
    logger.info("Smoke test answer: %r", answer)
    return answer
