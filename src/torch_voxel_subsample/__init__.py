from pathlib import Path

import torch
from torch import Tensor
from torch.utils import cpp_extension

src_dir = Path(__file__).parent
imp = cpp_extension.load(
    name='imp',
    sources=[str(src_dir / 'module.cpp')],
    extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
    # extra_cuda_cflags=["-Xptxas", "-v"]  # not necessary for now
)


def voxel_subsample(pos: Tensor, voxel_size: float, hash_size: float = 1.0, pick: int = None) -> Tensor:
    assert pos.ndim == 2 and pos.shape[1] == 3 and pos.dtype == torch.float
    if pos.stride(0) != 3:
        pos = pos.contiguous()

    def next_prime(x) -> int:
        r"""
        Finds the next prime, x included.
        x should be >= 3 for a correct result.
        """
        x = int(x) | 1
        for i in range(x, 2 * x, 2):
            prime = True
            for j in range(3, int(i ** 0.5) + 1, 2):
                if i % j == 0:
                    prime = False
                    break
            if prime:
                return i

    size = next_prime(pos.shape[0] * hash_size)  # todo: check this isn't too slow
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 3,), dtype=torch.int64)
    if pick is None:
        return imp.voxel_subsample(pos, voxel_size, table, storage)
    else:
        return imp.voxel_subsample_deterministic(pos, voxel_size, table, storage, pick)
