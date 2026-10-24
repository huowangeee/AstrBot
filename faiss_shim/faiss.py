"""faiss 的最小 numpy 兼容实现（AstrBot armv7 专用）。

为什么需要它
------------
``faiss-cpu`` 从来没有发布过 linux_armv7l 轮子（PyPI 与 piwheels 均无），源码编译
需要 BLAS/LAPACK 且内存开销巨大，玩客云（Cortex-A5 / 1GB）无法完成。

而 AstrBot 的 ``astrbot/core/db/vec_db/faiss_impl/embedding_storage.py`` 在 **导入期**
就 ``import faiss``，缺包会直接抛 ImportError。所以这里用 numpy 实现它真正用到的
那部分 API，让知识库/向量检索功能照常可用（检索算法为暴力 L2，数据量在几千条
量级时完全够用）。

已实现的接口（正好覆盖 AstrBot 的用法）
--------------------------------------
- ``normalize_L2(x)``                    原地归一化
- ``IndexFlatL2(d)`` / ``IndexFlatIP(d)``
- ``IndexIDMap(inner)`` / ``IndexIDMap2`` -> ``add_with_ids`` / ``search`` / ``ntotal``
- ``read_index(path)`` / ``write_index(index, path)``

未实现：IVF/PQ/HNSW 等索引类型、GPU、``remove_ids``、``reconstruct``。

索引文件格式：本 shim 自己 pickle，**与原生 faiss 的二进制格式不兼容**。
如果是从 amd64 机器迁移过来的 data 目录，请删除旧的向量索引文件重新建库。
"""

from __future__ import annotations

import os
import pickle

import numpy as np

__version__ = "shim-1.0"

_MAGIC = b"ASTRBOT-ARMV7-FAISS-SHIM-1"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def normalize_L2(x):
    """就地做 L2 归一化，行为对齐 ``faiss.normalize_L2``。"""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr /= norm
    else:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr /= norms
    return arr


# ---------------------------------------------------------------------------
# 基础索引
# ---------------------------------------------------------------------------
class _FlatIndex:
    metric = "l2"

    def __init__(self, d: int):
        self.d = int(d)
        self._vecs = np.zeros((0, self.d), dtype=np.float32)

    # -- 属性 --------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        return int(self._vecs.shape[0])

    def is_trained(self) -> bool:  # 兼容少量调用方
        return True

    # -- 写入 --------------------------------------------------------------
    def add(self, x):
        arr = np.ascontiguousarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.d:
            raise ValueError(f"维度不匹配: 期望 {self.d}, 实际 {arr.shape[1]}")
        self._vecs = arr.copy() if self.ntotal == 0 else np.vstack([self._vecs, arr])

    def add_with_ids(self, x, ids):
        raise RuntimeError("add_with_ids 只能在 IndexIDMap 上调用")

    def reset(self):
        self._vecs = np.zeros((0, self.d), dtype=np.float32)

    # -- 检索 --------------------------------------------------------------
    def _distances(self, queries: np.ndarray) -> np.ndarray:
        if self.ntotal == 0:
            return np.zeros((queries.shape[0], 0), dtype=np.float32)
        if self.metric == "l2":
            # ||q||^2 - 2q·x + ||x||^2
            qq = (queries * queries).sum(axis=1)[:, None]
            xx = (self._vecs * self._vecs).sum(axis=1)[None, :]
            dist = qq - 2.0 * (queries @ self._vecs.T) + xx
            return np.maximum(dist, 0.0).astype(np.float32)
        return (queries @ self._vecs.T).astype(np.float32)

    def _topk(self, dists: np.ndarray, k: int):
        n, m = dists.shape
        k = max(0, int(k))
        # 与 faiss 一致：始终返回 (n, k)，不足的位置用 -1 / inf 填充
        D = np.full((n, k), np.inf, dtype=np.float32)
        I = np.full((n, k), -1, dtype=np.int64)
        if m == 0 or k == 0:
            return D, I
        take = min(k, m)
        # L2 取最小、IP 取最大；统一转成"越大越好"再排序
        score = -dists if self.metric == "l2" else dists
        if take < m:
            part = np.argpartition(-score, take - 1, axis=1)[:, :take]
        else:
            part = np.tile(np.arange(m), (n, 1))
        part_score = np.take_along_axis(score, part, axis=1)
        order = np.argsort(-part_score, axis=1)
        picked = np.take_along_axis(part, order, axis=1)
        D[:, :take] = np.take_along_axis(dists, picked, axis=1)
        I[:, :take] = picked.astype(np.int64)
        return D, I

    def search(self, x, k):
        queries = np.ascontiguousarray(x, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        return self._topk(self._distances(queries), k)


class IndexFlatL2(_FlatIndex):
    metric = "l2"


class IndexFlatIP(_FlatIndex):
    metric = "ip"


class IndexIDMap:
    """把 ID 映射到内层索引，语义对齐 faiss.IndexIDMap。"""

    def __init__(self, index: _FlatIndex):
        self.index = index
        self._ids = np.zeros((0,), dtype=np.int64)

    @property
    def d(self) -> int:
        return self.index.d

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def is_trained(self) -> bool:
        return True

    def add_with_ids(self, x, ids):
        arr = np.ascontiguousarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        new_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if new_ids.shape[0] != arr.shape[0]:
            raise ValueError("向量数量与 id 数量不一致")
        # 已存在的 id 先移除，保持"一个 id 一份向量"
        if self.ntotal:
            keep = ~np.isin(self._ids, new_ids)
            if not keep.all():
                self.index._vecs = self.index._vecs[keep]
                self._ids = self._ids[keep]
        self.index.add(arr)
        self._ids = np.concatenate([self._ids, new_ids])

    def reset(self):
        self.index.reset()
        self._ids = np.zeros((0,), dtype=np.int64)

    def search(self, x, k):
        D, I = self.index.search(x, k)
        real = I >= 0
        out = np.full(I.shape, -1, dtype=np.int64)
        out[real] = self._ids[I[real]]
        return D, out


IndexIDMap2 = IndexIDMap


# ---------------------------------------------------------------------------
# 序列化（自有格式，与原生 faiss 不兼容）
# ---------------------------------------------------------------------------
def _snapshot(index):
    if isinstance(index, IndexIDMap):
        return {
            "wrapped": True,
            "metric": index.index.metric,
            "d": index.index.d,
            "vecs": index.index._vecs,
            "ids": index._ids,
        }
    if isinstance(index, _FlatIndex):
        return {
            "wrapped": False,
            "metric": index.metric,
            "d": index.d,
            "vecs": index._vecs,
            "ids": None,
        }
    raise TypeError(f"不支持序列化的索引类型: {type(index)!r}")


def write_index(index, path):
    data = _snapshot(index)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fp:
        fp.write(_MAGIC)
        pickle.dump(data, fp, protocol=4)
    os.replace(tmp, path)


def read_index(path):
    with open(path, "rb") as fp:
        magic = fp.read(len(_MAGIC))
        if magic != _MAGIC:
            raise RuntimeError(
                f"{path} 不是 armv7 faiss shim 写出的索引文件（可能是原生 faiss 格式），"
                "无法读取。请删除旧索引后重建知识库。"
            )
        data = pickle.load(fp)
    flat_cls = IndexFlatIP if data["metric"] == "ip" else IndexFlatL2
    flat = flat_cls(data["d"])
    flat._vecs = np.asarray(data["vecs"], dtype=np.float32)
    if not data.get("wrapped"):
        return flat
    wrapped = IndexIDMap(flat)
    wrapped._ids = np.asarray(data["ids"], dtype=np.int64)
    return wrapped
