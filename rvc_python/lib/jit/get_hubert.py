import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.utils import index_put

def pad_to_multiple(x, multiple, dim=-1, value=0):
    """Pads input tensor to a multiple of `multiple` along specified dimension."""
    if x is None:
        return None, 0
    tsz = x.size(dim)
    remainder = (math.ceil(tsz / multiple) * multiple) - tsz
    if remainder == 0:
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder

def extract_features(self, x, padding_mask=None, tgt_layer=None, min_layer=0):
    """Extracts features from the encoder with optimized padding and layer processing."""
    if padding_mask is not None:
        x = index_put(x, padding_mask, 0)

    x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
    x = x + x_conv

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)
    if pad_length > 0 and padding_mask is None:
        padding_mask = torch.zeros((x.size(0), x.size(1)), dtype=torch.bool, device=x.device)
        padding_mask[:, -pad_length:] = True
    else:
        padding_mask, _ = pad_to_multiple(padding_mask, self.required_seq_len_multiple, dim=-1, value=True)

    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x.transpose(0, 1)  # B x T x C -> T x B x C

    layer_results = []
    r = None
    for i, layer in enumerate(self.layers):
        if not self.training or random.random() > self.layerdrop:
            x, (z, lr) = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
            if i >= min_layer:
                layer_results.append((x, z, lr))
            if i == tgt_layer:
                r = x
                break

    if r is not None:
        x = r

    x = x.transpose(0, 1)  # T x B x C -> B x T x C
    if pad_length > 0:
        x = x[:, :-pad_length]
        layer_results = [(lr[0][:-pad_length], lr[1][:-pad_length] if lr[1] is not None else None, lr[2][:-pad_length]) for lr in layer_results]

    return x, layer_results

def compute_mask_indices(
    shape, padding_mask, mask_prob, mask_length, mask_type="static", 
    mask_other=0.0, min_masks=0, no_overlap=False, min_space=0, 
    require_same_masks=True, mask_dropout=0.0
):
    """Computes random mask spans for a given shape with optimized logic."""
    bsz, all_sz = shape
    mask = torch.full((bsz, all_sz), False, device=padding_mask.device if padding_mask is not None else torch.device('cpu'))

    all_num_mask = max(min_masks, int(mask_prob * all_sz / mask_length + torch.rand(1).item()))

    mask_idcs = []
    for i in range(bsz):
        sz = all_sz - padding_mask[i].long().sum().item() if padding_mask is not None else all_sz
        num_mask = max(min_masks, int(mask_prob * sz / mask_length + np.random.rand()))

        if mask_type == "static":
            lengths = torch.full((num_mask,), mask_length, dtype=torch.int)
        elif mask_type == "uniform":
            lengths = torch.randint(int(mask_other), mask_length * 2 + 1, (num_mask,))
        elif mask_type == "normal":
            lengths = torch.normal(mask_length, mask_other, size=(num_mask,)).int().clamp(min=1)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

        if lengths.sum() == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []
            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                valid_parts = [(s, e) for s, e in parts if e - s >= length + min_space]
                if not valid_parts:
                    break
                probs = torch.tensor([e - s for s, e in valid_parts], dtype=torch.float)
                c = torch.multinomial(probs / probs.sum(), 1).item()
                s, e = valid_parts[c]
                span_start = torch.randint(s, e - length, (1,)).item()
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - min_space >= min_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > min_length:
                    new_parts.append((span_start + length + min_space, e))
                parts = new_parts
            mask_idc = torch.tensor(mask_idc, dtype=torch.long)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            mask_idc = torch.tensor(random.sample(range(sz - min_len), num_mask))
            mask_idc = torch.cat([mask_idc[j] + torch.arange(lengths[j]) for j in range(len(mask_idc))])

        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min(len(m) for m in mask_idcs)
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = torch.tensor(random.sample(list(mask_idc), min_len))
        if mask_dropout > 0:
            num_holes = int(round(len(mask_idc) * mask_dropout))
            mask_idc = torch.tensor(random.sample(list(mask_idc), len(mask_idc) - num_holes))
        mask[i, mask_idc] = True

    return mask

def apply_mask(self, x, padding_mask, target_list):
    """Applies masking to input tensor for both time and channel dimensions."""
    B, T, C = x.shape
    if self.mask_prob > 0:
        mask_indices = compute_mask_indices(
            (B, T), padding_mask, self.mask_prob, self.mask_length,
            self.mask_selection, self.mask_other, min_masks=2,
            no_overlap=self.no_mask_overlap, min_space=self.mask_min_space
        ).to(x.device)
        x[mask_indices] = self.mask_emb

    if self.mask_channel_prob > 0:
        mask_channel_indices = compute_mask_indices(
            (B, C), None, self.mask_channel_prob, self.mask_channel_length,
            self.mask_channel_selection, self.mask_channel_other,
            no_overlap=self.no_mask_channel_overlap, min_space=self.mask_channel_min_space
        ).to(x.device).unsqueeze(1).expand(-1, T, -1)
        x[mask_channel_indices] = 0

    return x, mask_indices

def get_hubert_model(model_path="base_model/hubert_base.pt"):
    """Loads HuBERT model and assigns device based on GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, _, _ = load_model_ensemble_and_task([model_path], suffix="")
    hubert_model = models[0].to(device)

    hubert_model.apply_mask = lambda x, padding_mask, target_list: apply_mask(hubert_model, x, padding_mask, target_list)
    hubert_model.encoder.extract_features = lambda x, padding_mask=None, tgt_layer=None, min_layer=0: extract_features(
        hubert_model.encoder, x, padding_mask, tgt_layer, min_layer
    )

    def hubert_extract_features(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        res = self._forward(source, padding_mask=padding_mask, mask=mask, features_only=True, output_layer=output_layer)
        return res["features"] if ret_conv else res["x"], res["padding_mask"]

    hubert_model.extract_features = lambda source, padding_mask=None, mask=False, ret_conv=False, output_layer=None: hubert_extract_features(
        hubert_model, source, padding_mask, mask, ret_conv, output_layer
    )

    def infer(source, padding_mask, output_layer):
        output_layer = output_layer.item()
        logits, padding_mask = hubert_model.extract_features(source, padding_mask, output_layer=output_layer)
        return hubert_model.final_proj(logits) if output_layer == 9 else logits

    hubert_model.infer = infer
    return hubert_model
