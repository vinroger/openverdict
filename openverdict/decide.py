"""Encode a question and answer it in either mode. Shared by inference, training and evaluation, so all three see the same inputs.

Example:
    enc = Encoder(processor)
    item = enc.encode(case, case['questions'][0]['id'], thinking=True)
    out = think(backbone, enc, item, budget=80)        # {'cot': 'Leda=T. Shalonda=F. ...', 'probabilities': {...}, 'truncated': False}
"""
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch

from .format import parse_question
from .model import hidden_states, option_logits
from .prompts import ANSWER_SUFFIX, build_codebook, chat_prompt, render_question

DEFAULT_MAX_PIXELS = 262_144
DEFAULT_MIN_PIXELS = 65_536


@dataclass
class Encoded:
    inputs: dict            # tensors for the model, batch size 1
    code_ids: list          # token id of each option's code, in option order
    question: object        # the parsed Question
    prompt: str


class Encoder:
    def __init__(self, processor, *, image_root='.', max_tokens=8192, max_pixels=DEFAULT_MAX_PIXELS):
        self.processor = processor
        self.tokenizer = getattr(processor, 'tokenizer', processor)
        self.codebook = build_codebook(self.tokenizer)
        self.suffix_ids = self.tokenizer.encode(ANSWER_SUFFIX, add_special_tokens=False)
        self.stop_id = self.tokenizer.convert_tokens_to_ids('</think>')
        if self.suffix_ids[0] != self.stop_id:
            raise ValueError('"</think>" must be a single token at the start of the answer suffix')
        self.image_root, self.max_tokens, self.max_pixels = Path(image_root), max_tokens, max_pixels

    def encode(self, case, question_id, *, thinking, require_target=False, instruction=None):
        raw = next(q for q in case['questions'] if q['id'] == question_id)
        question = parse_question(raw, require_target=require_target)
        entries = self.codebook[:len(question.option_ids)]
        text = render_question(case, question, [c for c, _ in entries])
        images = [Image.open(self.image_root / ref['path']).convert('RGB') for ref in case.get('images') or []]
        prompt = chat_prompt(self.processor, text, len(images), thinking=thinking, instruction=instruction)
        if images:
            pixels = case['images'][0].get('max_pixels', self.max_pixels)
            inputs = self.processor(text=prompt, images=images, return_tensors='pt',
                                    images_kwargs={'size': {'shortest_edge': DEFAULT_MIN_PIXELS, 'longest_edge': pixels}})
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        inputs = {k: v for k, v in dict(inputs).items()
                  if k in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw', 'mm_token_type_ids')}
        if inputs['input_ids'].shape[1] > self.max_tokens:
            raise ValueError(f'input is longer than {self.max_tokens} tokens (inputs are never truncated)')
        return Encoded(inputs, [t for _, t in entries], question, prompt)


def to_device(inputs, device):
    return {k: v.to(device) for k, v in inputs.items()}


def append_tokens(inputs, ids):
    """The same inputs with extra text tokens at the end (a CoT and the answer suffix)."""
    extra = torch.as_tensor(ids, dtype=inputs['input_ids'].dtype, device=inputs['input_ids'].device).view(1, -1)
    out = dict(inputs)
    out['input_ids'] = torch.cat([inputs['input_ids'], extra], 1)
    out['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(extra)], 1)
    if 'mm_token_type_ids' in inputs:
        out['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(extra)], 1)
    return out


def answer_logits(backbone, item, cot_ids=None, suffix_ids=None):
    """Option logits of one question. Non-thinking: read at the end of the prompt. Thinking: append the CoT and "</think>\\n\\n" first."""
    device = backbone.get_output_embeddings().weight.device
    inputs = to_device(item.inputs, device)
    if cot_ids is not None:
        inputs = append_tokens(inputs, [*cot_ids, *suffix_ids])
    h = hidden_states(backbone, inputs)
    return option_logits(backbone, h[0, -1], item.code_ids)


def split_generation(ids, *, stop_id, eos_ids, budget):
    """Classify generated tokens. Example: [t1, t2, </think>] -> CoT [t1, t2], finished; 80 tokens with no stop -> truncated."""
    end = next((i for i, t in enumerate(ids) if t == stop_id or t in eos_ids), None)
    if end is not None:
        return {'cot_ids': list(ids[:end]), 'generated_ids': list(ids[:end + 1]), 'truncated': False,
                'valid': ids[end] == stop_id}
    if len(ids) != budget:
        raise ValueError('generation stopped early without a stop token')
    return {'cot_ids': list(ids), 'generated_ids': list(ids), 'truncated': True, 'valid': True}


@torch.no_grad()
def generate_cots(backbone, encoder, item, *, budget=80, samples=1, temperature=None, top_p=1.0, top_k=0, blocked_ids=()):
    """Write `samples` CoTs for one question. Greedy when temperature is None (inference), sampled otherwise (training)."""
    device = backbone.get_output_embeddings().weight.device
    inputs = to_device(item.inputs, device)
    eos = backbone.generation_config.eos_token_id
    eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    kwargs = dict(max_new_tokens=budget, eos_token_id=[encoder.stop_id, *eos], pad_token_id=encoder.tokenizer.pad_token_id,
                  num_return_sequences=samples, use_cache=True)
    if temperature is None:
        kwargs.update(do_sample=False)
    else:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        if blocked_ids:
            kwargs.update(suppress_tokens=list(blocked_ids))
    backbone.eval()
    sequences = backbone.generate(**inputs, **kwargs)
    start = inputs['input_ids'].shape[1]
    out = []
    for seq in sequences:
        ids = seq[start:].tolist()
        stop = next((i + 1 for i, t in enumerate(ids) if t == encoder.stop_id or t in eos), len(ids))
        out.append(split_generation(ids[:stop], stop_id=encoder.stop_id, eos_ids=eos, budget=budget))
    return out


def probabilities(logits, question):
    p = torch.softmax(logits.float(), -1).tolist()
    return dict(zip(question.option_ids, p))


@torch.no_grad()
def non_thinking(backbone, item):
    logits = answer_logits(backbone, item)
    return {'probabilities': probabilities(logits, item.question), 'logits': logits.cpu()}


@torch.no_grad()
def think(backbone, encoder, item, *, budget=80):
    """Thinking mode: a greedy CoT of at most `budget` tokens, then one pass that scores every option."""
    gen = generate_cots(backbone, encoder, item, budget=budget)[0]
    logits = answer_logits(backbone, item, gen['cot_ids'], encoder.suffix_ids)
    return {'cot': encoder.tokenizer.decode(gen['cot_ids'], skip_special_tokens=False), 'cot_tokens': len(gen['cot_ids']),
            'truncated': gen['truncated'], 'probabilities': probabilities(logits, item.question), 'logits': logits.cpu()}
