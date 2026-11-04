"""CPU-only fixtures for testing NumPy parts without importing GPU frameworks."""
import ast
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def delta_functions():
    path = Path(__file__).resolve().parents[1] / 'scripts/train_bfa_delta_diffusion.py'
    tree = ast.parse(path.read_text())
    names = {'frame_delta', 'reconstruct', 'anchor_features'}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'np': np, 'MOD': np.asarray([512, 512, 128, 128], dtype=np.int32)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace
