import taichi as ti
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.sparse.linalg import cg

@ti.data_oriented
class SparseMatrix:
    def __init__(self, n, use_cg=False, device="cuda:0"):
        self.n = n
        self.timestamp = ti.field(ti.i32, shape=())
        self.active_flag = ti.field(ti.i32, shape=(n, n))
        self.value = ti.field(ti.f64, shape=(n, n))
        self.row_cnt = ti.field(ti.i32, shape=(n,))
        self.row_idx = ti.field(ti.i32, shape=(n, n))
        self.use_cg = use_cg
        self.device = device

    @ti.kernel
    def clear_all(self):
        self.timestamp[None] += 1
        for i in range(self.n):
            for _j in range(self.row_cnt[i]):
                j = self.row_idx[i, _j]
                self.value[i, j] = 0
        for i in range(self.n):
            self.row_cnt[i] = 0

    @ti.func
    def add(self, i, j, v):
        timestamp = self.timestamp[None]
        self.value[i, j] += v
        t0 = ti.atomic_max(self.active_flag[i, j], timestamp)
        if t0 != timestamp:
            self.active_flag[i, j] = timestamp
            self.row_idx[i, ti.atomic_add(self.row_cnt[i], 1)] = j

    @ti.func
    def sub(self, i, j, v):
        self.add(i, j, -v)
    
    @ti.kernel
    def to_ndarray(self, arr:ti.types.ndarray()):
        for i in range(self.n):
            for _j in range(self.row_cnt[i]):
                j = self.row_idx[i, _j]
                arr[i, j] = self.value[i, j]

    @ti.kernel
    def check_nan(self):
        has_nan = False
        for i in range(self.n):
            for j in range(self.n):
                # j = self.row_idx[i, _j]
                if ti.math.isnan(self.value[i, j]):
                    has_nan = True
        if has_nan:
            print("nan in matrix!!!")

    def check_PD(self):
        H = self.value.to_torch().double()
        L, V = torch.linalg.eigh(H.cpu())
        if L.min() < -1e-5:
            print("not PD")
            return False
        return True

    @ti.kernel
    def fill_indptr(self, indptr:ti.types.ndarray()):
        ti.loop_config(serialize=True)
        indptr[0] = 0
        for i in range(self.n):
            indptr[i+1] = indptr[i] + self.row_cnt[i]

    @ti.kernel
    def fill_value(self, indptr:ti.types.ndarray(), ind:ti.types.ndarray(), v:ti.types.ndarray()):
        for i in range(self.n):
            offset = indptr[i]
            for j in range(self.row_cnt[i]):
                ind[offset+j] = j1 = self.row_idx[i, j]
                v[offset+j] = self.value[i, j1]

    def solve(self, b):
        indptr = torch.zeros((self.n+1,), device=self.device, dtype=torch.int32)
        self.fill_indptr(indptr)
        nzz = indptr[-1]
        ind = torch.zeros((nzz,), device=self.device, dtype=torch.int32)
        v = torch.zeros((nzz,), device=self.device, dtype=torch.float64)
        self.fill_value(indptr, ind, v)
        indptr = cp.from_dlpack(to_dlpack(indptr))
        ind = cp.from_dlpack(to_dlpack(ind))
        v = cp.from_dlpack(to_dlpack(v))
        b = cp.from_dlpack(to_dlpack(b.double()))
        H = csr_matrix((v, ind, indptr), shape=(self.n, self.n))
        # print('solve: n = %d, nzz = %d' % (self.n, int(nzz)))
        if self.use_cg:
            x0 = cp.zeros_like(b)
            x, info = cg(H, b, x0, tol=1e-6)
            x = from_dlpack(x.toDlpack())
        else:
            x = from_dlpack(spsolve(H, b).toDlpack())
        # print('done')
        return x

    def build(self):
        # self.check_PD()
        builder = ti.linalg.SparseMatrixBuilder(self.n, self.n, 10000000)
        self.to_taichi_sparse_builder(builder)
        return builder.build()

    # def solve(self, b, tol=1e-6, max_iter=1000):
    #
    #     indptr = torch.zeros((self.n + 1,), device='cuda:0', dtype=torch.int32)
    #     self.fill_indptr(indptr)
    #     nzz = indptr[-1]
    #     ind = torch.zeros((nzz,), device='cuda:0', dtype=torch.int32)
    #     v = torch.zeros((nzz,), device='cuda:0', dtype=torch.float64)
    #     self.fill_value(indptr, ind, v)
    #     indptr = cp.from_dlpack(to_dlpack(indptr))
    #     ind = cp.from_dlpack(to_dlpack(ind))
    #     v = cp.from_dlpack(to_dlpack(v))
    #     b = cp.from_dlpack(to_dlpack(b.double()))
    #     H = csr_matrix((v, ind, indptr), shape=(self.n, self.n))
    #
    #     x0 = cp.zeros_like(b)
    #     x = x0.copy()
    #     r = b - H.dot(x)
    #     p = r.copy()
    #     r_dot_r = cp.dot(r, r)
    #
    #     for i in range(max_iter):
    #         if r_dot_r < tol:
    #             break
    #         Hp = H.dot(p)
    #         alpha = r_dot_r / cp.dot(p, Hp)
    #         x += alpha * p
    #         r -= alpha * Hp
    #
    #         new_r_dot_r = cp.dot(r, r)
    #         beta = new_r_dot_r / r_dot_r
    #         p = r + beta * p
    #         r_dot_r = new_r_dot_r
    #
    #     x = from_dlpack(x.toDlpack())
    #     return x