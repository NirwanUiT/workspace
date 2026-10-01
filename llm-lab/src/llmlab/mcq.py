"""Zero-shot option scoring for multiple-choice questions.

One forward pass per question: build a prompt that ends right where the model
should write the answer letter, read the next-token logits, and keep only the
letter tokens. Each letter can be tokenised as "A" or " A"; we logsumexp the
variants into one logit per option, so a 4-way softmax over those logits is the
model's distribution over options. `letter_mass` records how much of the full
next-token distribution falls on the letter tokens at all (near 1 means the
model is answering in the expected format).
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

LETTERS = "ABCDEFGHIJ"


def format_question(question: str, choices: list[str], subject: str | None = None) -> str:
    head = ""
    if subject:
        head = f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{head}{question.strip()}\n{opts}"


def build_prompt(tokenizer, question, choices, subject=None, chat: bool = True) -> str:
    """Prompt ending exactly where the answer letter goes.

    chat=True: the model's chat template with thinking disabled, and the
    assistant turn pre-filled with "Answer:". chat=False: plain completion
    format ending in "Answer:" (the usual MMLU format for base models).
    """
    body = format_question(question, choices, subject)
    if not chat:
        return body + "\nAnswer:"
    msgs = [{"role": "user", "content": body + "\n\nAnswer with the letter of the correct option only."}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return text + "Answer:"


def letter_token_ids(tokenizer, k: int) -> list[list[int]]:
    """For each of the first k letters, the single-token ids of "X" and " X"."""
    out = []
    for letter in LETTERS[:k]:
        ids = set()
        for variant in (letter, " " + letter):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        if not ids:
            raise ValueError(f"letter {letter!r} has no single-token form")
        out.append(sorted(ids))
    return out


@torch.no_grad()
def score_prompts(model, tokenizer, prompts: list[str], k: int, batch_size: int = 16,
                  device: str | torch.device = "cuda", desc: str = "scoring"):
    """Return (option_logits (N, k), letter_mass (N,)) in prompt order.

    Prompts are sorted by length and left-padded so the last position of every
    row is the real last token.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    groups = letter_token_ids(tokenizer, k)
    flat = torch.tensor([t for g in groups for t in g], device=device)
    owner = torch.tensor([i for i, g in enumerate(groups) for _ in g], device=device)

    lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    order = np.argsort(lengths)[::-1]
    option_logits = np.zeros((len(prompts), k), dtype=np.float64)
    letter_mass = np.zeros(len(prompts), dtype=np.float64)

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, logits_to_keep=1)
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        sel = logp[:, flat]
        # logsumexp over the variants of each letter
        per_opt = torch.stack(
            [torch.logsumexp(sel[:, owner == j], dim=1) for j in range(k)], dim=1
        )
        option_logits[idx] = per_opt.double().cpu().numpy()
        letter_mass[idx] = torch.logsumexp(sel, dim=1).exp().double().cpu().numpy()
    return option_logits, letter_mass
