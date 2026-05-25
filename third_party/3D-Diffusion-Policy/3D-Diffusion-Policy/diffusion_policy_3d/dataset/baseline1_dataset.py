"""Baseline1 (Human Retarget DP) dataset adapter for DP3.

Same schema as realdex_dataset but without the `img` key — Baseline1 zarr only
has `point_cloud / state / action`. Point cloud width is 4096 (vs realdex's
1024); state/action dim is 8 ([x,y,z, qw,qx,qy,qz, gripper]).

Two storage modes:
  * lazy=False (default): copy_from_path → full numpy arrays in RAM. Fast per-sample
    access but needs ~2× zarr size in RAM. OK for DexYCB-only (14G zarr → ~16G RAM).
  * lazy=True: create_from_path → on-disk zarr accessed chunk-by-chunk. Slower
    per-sample but throughput is restored by num_workers>=4 + zarr's chunk cache.
    The normalizer is overridden to stream stats in chunks so it doesn't
    materialize the full point_cloud either.

Augmentation (yaw_aug=True): each sampled sequence is rigidly rotated about the
gravity axis (+Z_G) by a random angle — an exact task symmetry, since gravity,
the table and contact geometry are all unchanged by a yaw rotation. One angle
per sequence, shared across all T frames. This fills the 30-90 deg yaw holes in
DexYCB's object placement. The 'limits' normalizer is widened to match: the x/y
limits become +/-(max xy-radius) and the quaternion limits become [-1, 1], so
augmented samples still normalize into [-1, 1].
"""
from typing import Dict, Tuple
import copy
import numpy as np
import torch
import zarr

from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


def _stream_stats(src, chunk_rows: int = 1000
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Match _fit(last_n_dims=1): reshape to (-1, last_dim), stats over axis=0.

    Streams `src` in chunks of `chunk_rows` along axis 0 so memory stays at one
    chunk's worth (~tens of MB). Uses float64 accumulators for numerical
    stability, returns float32 to match LinearNormalizer's dtype.

    Also returns `r_xy_max` = max sqrt(x^2 + y^2) over the first two columns —
    the radius the yaw augmentation can rotate an x/y pair onto, used to widen
    the 'limits' normalizer so augmented samples stay inside [-1, 1].
    """
    last_dim = src.shape[-1]
    n_rows = src.shape[0]
    mn = np.full(last_dim, np.inf, dtype=np.float64)
    mx = np.full(last_dim, -np.inf, dtype=np.float64)
    mean = np.zeros(last_dim, dtype=np.float64)
    M2 = np.zeros(last_dim, dtype=np.float64)
    r2_max = 0.0
    count = 0
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        chunk = np.asarray(src[start:end]).reshape(-1, last_dim).astype(np.float64)
        n2 = chunk.shape[0]
        if n2 == 0:
            continue
        mn = np.minimum(mn, chunk.min(axis=0))
        mx = np.maximum(mx, chunk.max(axis=0))
        if last_dim >= 2:
            r2_max = max(r2_max, float((chunk[:, 0] ** 2 + chunk[:, 1] ** 2).max()))
        # parallel-algorithm Welford: combine chunk stats with running stats
        chunk_mean = chunk.mean(axis=0)
        chunk_M2 = ((chunk - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - mean
        new_count = count + n2
        mean = mean + delta * (n2 / new_count)
        M2 = M2 + chunk_M2 + (delta ** 2) * (count * n2 / new_count)
        count = new_count
    var = M2 / max(count, 1)
    return (mn.astype(np.float32), mx.astype(np.float32),
            mean.astype(np.float32), np.sqrt(var).astype(np.float32),
            float(np.sqrt(r2_max)))


def _limits_params(mn, mx, mean, std,
                   output_min: float = -1., output_max: float = 1.,
                   range_eps: float = 1e-4):
    """Mirror _fit()'s mode='limits' + fit_offset=True branch."""
    mn_t = torch.from_numpy(mn)
    mx_t = torch.from_numpy(mx)
    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)
    input_range = (mx_t - mn_t).clone()
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * mn_t
    offset[ignore_dim] = (output_max + output_min) / 2 - mn_t[ignore_dim]
    return scale, offset, dict(min=mn_t, max=mx_t, mean=mean_t, std=std_t)


class Baseline1Dataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 task_name=None,
                 lazy=False,
                 yaw_aug=False):
        super().__init__()
        self.task_name = task_name
        self.lazy = lazy
        self.yaw_aug = yaw_aug
        if lazy:
            self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        else:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        val_set.yaw_aug = False   # validation uses un-augmented poses → stable val_loss
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        assert mode == 'limits', "streaming normalizer only supports mode='limits'"
        norm = LinearNormalizer()
        # (norm_key, replay_buffer_key, chunk_rows)
        # point_cloud chunk_rows=500 → ~6 MB read at a time after reshape
        # state/action are tiny (8 cols), 100k rows = a few MB
        for nk, rk, cr in [('action',      'action',      100_000),
                           ('agent_pos',   'state',       100_000),
                           ('point_cloud', 'point_cloud',    500)]:
            src = self.replay_buffer[rk]
            mn, mx, mean, std, r_xy = _stream_stats(src, chunk_rows=cr)
            if self.yaw_aug:
                # yaw aug rotates the x/y pair onto a circle of radius up to r_xy,
                # so widen those two limits to +/-r_xy; the quaternion (dims 3:7)
                # gets re-mixed by the yaw rotation → bound it by the always-valid
                # [-1, 1]. z (dim 2) and gripper (dim 7) are untouched by yaw.
                mn, mx = mn.copy(), mx.copy()
                mn[0] = mn[1] = -r_xy
                mx[0] = mx[1] = r_xy
                if mn.shape[0] == 8:                 # state / action: quaternion dims
                    mn[3:7] = -1.0
                    mx[3:7] = 1.0
            scale, offset, stats = _limits_params(mn, mx, mean, std)
            norm[nk] = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
        return norm

    def __len__(self) -> int:
        return len(self.sampler)

    @staticmethod
    def _aug_yaw(sample: dict) -> dict:
        """Rigidly rotate the whole sampled sequence about the gravity axis (+Z_G)
        by a random angle — an exact task symmetry. One angle for all T frames:
        x/y are rotated, z and gripper are untouched, and the EE quaternion (wxyz)
        is pre-multiplied by the yaw quaternion. A fresh OS-entropy RNG is used so
        DataLoader workers don't share a seed (the classic numpy-in-Dataset bug).
        """
        theta = np.random.default_rng().uniform(0.0, 2.0 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        Rt = np.array([[c, s], [-s, c]], dtype=np.float32)    # row-vec @ Rt == R(theta) @ vec
        ch, sh = np.cos(theta / 2.0), np.sin(theta / 2.0)     # yaw quaternion half-angle
        out = dict(sample)
        pc = np.asarray(sample['point_cloud'], dtype=np.float32).copy()
        pc[..., :2] = pc[..., :2] @ Rt
        out['point_cloud'] = pc
        for k in ('state', 'action'):
            v = np.asarray(sample[k], dtype=np.float32).copy()
            v[:, :2] = v[:, :2] @ Rt
            w, x, y, z = (v[:, 3].copy(), v[:, 4].copy(),
                          v[:, 5].copy(), v[:, 6].copy())
            v[:, 3] = ch * w - sh * z          # q_z(theta) (x) q  in wxyz
            v[:, 4] = ch * x - sh * y
            v[:, 5] = ch * y + sh * x
            v[:, 6] = ch * z + sh * w
            out[k] = v
        return out

    def _sample_to_data(self, sample):
        agent_pos   = sample['state'][:, ].astype(np.float32)        # (T, 8)
        point_cloud = sample['point_cloud'][:, ].astype(np.float32)  # (T, 4096, 3)
        return {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos':   agent_pos,
            },
            'action': sample['action'].astype(np.float32),           # (T, 8)
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        if self.yaw_aug:
            sample = self._aug_yaw(sample)
        return dict_apply(self._sample_to_data(sample), torch.from_numpy)
