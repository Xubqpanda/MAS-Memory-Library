# src/registry/registry.py
from typing import Type

from src.solver.base import MetaSolver
from src.solver.autogen import AutoGen
from src.solver.macnet import MacNet
from src.solver.dylan import DyLAN
from src.solver.reasoning import ReasoningBase, ReasoningIO

from src.memory.base import SolverMemoryBase
from src.memory.methods import (
    EmptyMemory,
    GenerativeMASMemory,
    VoyagerMASMemory,
    MemoryBankMASMemory,
    ChatDevMASMemory,
    MetaGPTMASMemory,
    GMemory,
)


# ─── MAS  ────────────────────────────────────────────────────────────

MAS_REGISTRY: dict[str, Type[MetaSolver]] = {
    "autogen": AutoGen,
    "macnet": MacNet,
    "dylan": DyLAN,
}

# ─── Memory ─────────────────────────────────────────────────────────

MEMORY_REGISTRY: dict[str, Type[SolverMemoryBase]] = {
    "empty": EmptyMemory,
    "generative": GenerativeMASMemory,
    "voyager": VoyagerMASMemory,
    "memorybank": MemoryBankMASMemory,
    "chatdev": ChatDevMASMemory,
    "metagpt": MetaGPTMASMemory,
    "g-memory": GMemory,
}

# ─── Reasoning  ──────────────────────────────────────────────────────

REASONING_REGISTRY: dict[str, Type[ReasoningBase]] = {
    "io": ReasoningIO,
}


# ─── get functions ──────────────────────────────────────────────────────────────────

def get_mas_cls(name: str) -> Type[MetaSolver]:
    if name not in MAS_REGISTRY:
        raise ValueError(f"Unknown MAS framework '{name}'. Available: {list(MAS_REGISTRY)}")
    return MAS_REGISTRY[name]


def get_memory_cls(name: str) -> Type[SolverMemoryBase]:
    if name not in MEMORY_REGISTRY:
        raise ValueError(f"Unknown memory method '{name}'. Available: {list(MEMORY_REGISTRY)}")
    return MEMORY_REGISTRY[name]


def get_reasoning_cls(name: str) -> Type[ReasoningBase]:
    if name not in REASONING_REGISTRY:
        raise ValueError(f"Unknown reasoning type '{name}'. Available: {list(REASONING_REGISTRY)}")
    return REASONING_REGISTRY[name]