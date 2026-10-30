# SPDX-License-Identifier: Apache-2.0
"""Tensor-only inference for the pinned two-layer 1.5M checkpoint.

All eight experts are evaluated, with the learned top-2 gate preserved. This
trades sparse compute for a small, portable graph. No Triton or custom ONNX ops.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


def parameter(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value.clone(), requires_grad=False)


def linear(weights: dict, name: str) -> nn.Linear:
    w = weights[name + ".weight"]
    layer = nn.Linear(w.shape[1], w.shape[0], bias=name + ".bias" in weights)
    layer.weight = parameter(w)
    if layer.bias is not None:
        layer.bias = parameter(weights[name + ".bias"])
    return layer


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = parameter(weight)
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps))


class Expert(nn.Module):
    def __init__(self, up, gate, down):
        super().__init__()
        self.up = nn.Linear(up.shape[0], up.shape[1], bias=False)
        self.gate = nn.Linear(gate.shape[0], gate.shape[1], bias=False)
        self.down = nn.Linear(down.shape[0], down.shape[1], bias=False)
        self.up.weight = parameter(up.T.contiguous())
        self.gate.weight = parameter(gate.T.contiguous())
        self.down.weight = parameter(down.T.contiguous())

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, w, prefix, config):
        super().__init__()
        self.norm = RMSNorm(w[prefix + ".norm.weight"], config["norm_eps"])
        self.q = linear(w, prefix + ".Q_proj")
        self.k = linear(w, prefix + ".K_proj")
        self.v = linear(w, prefix + ".V_proj")
        self.out = linear(w, prefix + ".O_proj")
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["hidden_size"] // self.heads
        inv = 1.0 / config["rope_theta"] ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        angles = torch.arange(512, dtype=torch.float32)[:, None] * inv[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def rotate(self, x, sequence):
        left, right = x.chunk(2, dim=-1)
        return x * self.cos[:sequence] + torch.cat([-right, left], -1) * self.sin[:sequence]

    def forward(self, x, mask):
        batch, sequence, hidden = x.shape
        z = self.norm(x)
        q = self.q(z).reshape(batch, sequence, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(z).reshape(batch, sequence, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(z).reshape(batch, sequence, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotate(q, sequence), self.rotate(k, sequence)
        repeat = self.heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        logits = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        logits = torch.where(mask[:, None, None, :].bool(), logits, -torch.inf)
        z = (logits.softmax(-1) @ v).transpose(1, 2).reshape(batch, sequence, hidden)
        return x + self.out(z)


class Block(nn.Module):
    def __init__(self, w, prefix, config):
        super().__init__()
        self.attention = Attention(w, prefix + ".attention", config)
        self.norm = RMSNorm(w[prefix + ".ffn_norm.weight"], config["norm_eps"])
        fp = prefix + ".ffn"
        self.router = linear(w, fp + ".router")
        self.experts = nn.ModuleList(
            [
                Expert(w[fp + ".W_up"][e], w[fp + ".W_gate"][e], w[fp + ".W_down"][e])
                for e in range(config["num_routed_experts"])
            ]
        )
        self.shared_up = linear(w, fp + ".shared_up")
        self.shared_gate = linear(w, fp + ".shared_gate")
        self.shared_down = linear(w, fp + ".shared_down")

    def forward(self, x, mask):
        x = self.attention(x, mask)
        z = self.norm(x)
        probs = self.router(z).softmax(-1)
        values, indices = probs.topk(2, dim=-1)
        values = values / (values.sum(-1, keepdim=True) + 1e-9)
        gates = torch.zeros_like(probs).scatter(-1, indices, values)
        out = self.shared_down(F.gelu(self.shared_gate(z)) * self.shared_up(z))
        for e, expert in enumerate(self.experts):
            out = out + expert(z) * gates[..., e : e + 1]
        return x + out


class PortableEncoder(nn.Module):
    def __init__(self, source: Path):
        super().__init__()
        cfg = json.loads((source / "config.json").read_text())
        if (
            cfg["hidden_size"],
            cfg["num_hidden_layers"],
            cfg["hash_projections"],
            cfg["num_routed_experts"],
            cfg["top_k"],
            cfg["num_shared_experts"],
        ) != (128, 2, 16, 8, 2, 1):
            raise ValueError("Unsupported checkpoint architecture")
        w = load_file(source / "model.safetensors")
        table = w["embeddings.hash_embeddings.weight"]
        self.table = nn.Embedding(table.shape[0], table.shape[1])
        self.table.weight = parameter(table)
        self.embedding_projection = linear(w, "embeddings.projection")
        self.embedding_norm = nn.LayerNorm(128, eps=1e-5)
        self.embedding_norm.weight = parameter(w["embeddings.norm.weight"])
        self.embedding_norm.bias = parameter(w["embeddings.norm.bias"])
        self.blocks = nn.ModuleList([Block(w, f"layers.{i}", cfg) for i in range(2)])
        self.norm = RMSNorm(w["norm.weight"], cfg["norm_eps"])
        head = load_file(source / "1_Dense/model.safetensors")["linear.weight"]
        self.projection = nn.Linear(128, 128, bias=False)
        self.projection.weight = parameter(head)
        self.eval()

    def forward(self, input_ids, attention_mask):
        # Upstream explicitly overrides the query attention mask based on marker ID.
        query = input_ids[:, 1:2] == 30000
        mask = torch.where(query, torch.ones_like(attention_mask), attention_mask)
        x = self.embedding_norm(self.embedding_projection(self.table(input_ids)))
        for block in self.blocks:
            x = block(x, mask)
        return F.normalize(self.projection(self.norm(x)), p=2, dim=-1, eps=1e-12)
