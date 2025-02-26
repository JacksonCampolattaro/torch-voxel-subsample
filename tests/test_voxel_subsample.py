import math
import time

import pytest
import torch

from torch_voxel_subsample import voxel_subsample


def test_subsampling_fully_occupied():
    num_cells = 4 ** 3
    voxel_size = 1 / math.cbrt(num_cells)

    # Generate a random dense point cloud
    pos = torch.rand(10_000, 3)

    # Sample it sparsely (ensuring that every cell will contain a particle)
    samples = voxel_subsample(pos, voxel_size)

    assert len(samples) == num_cells


def test_subsampling_random():
    pos = torch.rand(10_000, 3)
    samples_a = voxel_subsample(pos, 1 / 16)
    samples_b = voxel_subsample(pos, 1 / 16)

    assert len(samples_a) == len(samples_b)
    assert not (samples_a == samples_b).all()


def test_subsampling_deterministic():
    pos = torch.rand(10_000, 3)
    samples_a = voxel_subsample(pos, 1 / 16, pick=0)
    samples_b = voxel_subsample(pos, 1 / 16, pick=0)

    assert len(samples_a) == len(samples_b)
    assert (samples_a == samples_b).all()


@pytest.mark.parametrize('n', [100, 1_000, 10_000])
@pytest.mark.parametrize('voxel_size', [1 / 2, 1 / 8, 1 / 32])
def test_subsampling_correctness(n: int, voxel_size: float):
    # Generate a random point cloud
    pos = torch.rand(n, 3)

    # Apply sampling
    samples = voxel_subsample(pos, voxel_size)

    # todo: check spacing between particles

    num_voxels = (1 / voxel_size) ** 3
    assert len(samples) <= num_voxels


def test_subsampling_speed():
    pos = torch.rand(1_000_000, 3)
    start_time = time.time()
    samples = voxel_subsample(pos, 1 / 1_000)
    elapsed_time = time.time() - start_time
    assert len(samples) > 0
    print(f" ({elapsed_time:.02}s)")
    assert elapsed_time < 1.0
