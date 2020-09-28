import numpy as np
import torch

from mmpose.models import MeshHMRHead


def test_mesh_hmr_head():
    """Test hmr mesh head."""
    head = MeshHMRHead(in_channels=512)
    head.init_weights()

    input_shape = (1, 512, 8, 8)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    smpl_rotmat, smpl_shape, camera = out
    assert smpl_rotmat.shape == torch.Size([1, 24, 3, 3])
    assert smpl_shape.shape == torch.Size([1, 10])
    assert camera.shape == torch.Size([1, 3])
    """Test hmr mesh head."""
    head = MeshHMRHead(
        in_channels=512,
        smpl_mean_params='tests/data/smpl/smpl_mean_params.npz',
        n_iter=3)
    head.init_weights()
    input_shape = (1, 512, 8, 8)
    inputs = _demo_inputs(input_shape)
    out = head(inputs)
    smpl_rotmat, smpl_shape, camera = out
    assert smpl_rotmat.shape == torch.Size([1, 24, 3, 3])
    assert smpl_shape.shape == torch.Size([1, 10])
    assert camera.shape == torch.Size([1, 3])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run mesh head.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    Returns:
        Random input tensor with the size of input_shape.
    """
    inps = np.random.random(input_shape)
    inps = torch.FloatTensor(inps)
    return inps
