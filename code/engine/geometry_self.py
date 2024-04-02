import numpy as np
import taichi as ti
import taichi.math as tm
import matplotlib.pyplot as plt

vec3 = ti.types.vector(3, ti.f64)

grid_h = 0.1
grid_n = int(0.2 // grid_h) * 2
grid_bound = grid_h * (grid_n-1) / 2
max_n_particles = 100000

grid_cnt = ti.field(ti.i32, shape=(grid_n, grid_n, grid_n))
grid_scnt = ti.field(ti.i32, shape=(grid_n, grid_n))
grid_s2cnt = ti.field(ti.i32, shape=(grid_n,))
grid_baseidx = ti.field(ti.i32, shape=(grid_n, grid_n, grid_n))
grid_active_range = ti.Vector.field(3, ti.i32, shape=(2,))
particle_v = ti.Vector.field(3, ti.f64, shape=(max_n_particles, 3))
particle_idx = ti.field(ti.i32, shape=(max_n_particles, 3))

# proj_num = ti.field(ti.i32, shape=())

@ti.func
def pt2tri(x:tm.vec3, p1:tm.vec3, p2:tm.vec3, p3:tm.vec3):
    eps = 0
    e1 = (p2-p1).normalized()
    e2 = (p3-p2).normalized()
    e3 = (p1-p3).normalized()
    n = -e1.cross(e3).normalized()
    x1 = x - (x-p1).dot(n)*n
    d = 0.0
    c = 0
    w = vec3(0)
    if (x1-p1).cross(e1).dot(n) > eps:
        if (x1-p1).dot(e1) < -eps:
            c = 1 # p1
            d = (x-p1).norm()
            w = vec3(1, 0, 0)
        elif (x1-p2).dot(e1) > eps:
            c = 2 # p2
            d = (x-p2).norm()
            w = vec3(0, 1, 0)
        else:
            c = -3 # e12
            alpha = (x1-p1).dot(e1) / (p2-p1).dot(e1)
            x2 = p1 + alpha*(p2-p1)
            d = (x-x2).norm()
            w = vec3(1-alpha, alpha, 0)
    elif (x1-p2).cross(e2).dot(n) > eps:
        if (x1-p2).dot(e2) < -eps:
            c = 2 # p2
            d = (x-p2).norm()
            w = vec3(0, 1, 0)
        elif (x1-p3).dot(e2) > eps:
            c = 3 # p3
            d = (x-p3).norm()
            w = vec3(0, 0, 1)
        else:
            c = -1 # e23
            alpha = (x1-p2).dot(e2) / (p3-p2).dot(e2)
            x2 = p2 + alpha*(p3-p2)
            d = (x-x2).norm()
            w = vec3(0, 1-alpha, alpha)
    elif (x1-p3).cross(e3).dot(n) > eps:
        if (x1-p3).dot(e3) < -eps:
            c = 3 # p3
            d = (x-p3).norm()
            w = vec3(0, 0, 1)
        elif (x1-p1).dot(e3) > eps:
            c = 1 # p1
            d = (x-p1).norm()
            w = vec3(1, 0, 0)
        else:
            c = -2 # e31
            alpha = (x1-p3).dot(e3) / (p1-p3).dot(e3)
            x2 = p3 + alpha*(p1-p3)
            d = (x-x2).norm()
            w = vec3(alpha, 0, 1-alpha)
    else:
        # f123
        d = (x-x1).norm()
        S = (p3 - p1).cross(p2 - p1).norm()
        w1 = (p3 - p2).cross(x1 - p2).dot(n) / S
        w2 = (p1 - p3).cross(x1 - p3).dot(n) / S
        w3 = (p2 - p1).cross(x1 - p1).dot(n) / S
        w = vec3(w1, w2, w3)
    return c, d, w

@ti.func
def grid_idx(x: vec3):
    i = ti.floor(tm.clamp(x[0], -grid_bound, grid_bound) / grid_h, ti.i32) + grid_n // 2
    j = ti.floor(tm.clamp(x[1], -grid_bound, grid_bound) / grid_h, ti.i32) + grid_n // 2
    k = ti.floor(tm.clamp(x[2], -grid_bound, grid_bound) / grid_h, ti.i32) + grid_n // 2
    return i, j, k

@ti.kernel
def p2g(v:ti.template(), f:ti.template(), start: ti.i32, end: ti.i32):
    active_old = (
        (grid_active_range[0][0], grid_active_range[1][0]+1),
        (grid_active_range[0][1], grid_active_range[1][1]+1),
        (grid_active_range[0][2], grid_active_range[1][2]+1),
    )
    for i, j, k in ti.ndrange(active_old[0], active_old[1], active_old[2]):
        grid_cnt[i, j, k] = 0

    grid_active_range[0] = (grid_n, grid_n, grid_n)
    grid_active_range[1] = (0, 0, 0)
    for i in range(start, end):
        mid_v = (v[f[i][0]] + v[f[i][1]] + v[f[i][2]])/3
        idx = grid_idx(mid_v)
        grid_cnt[idx] += 1
        ti.atomic_min(grid_active_range[0], idx)
        ti.atomic_max(grid_active_range[1], idx)

    active = (
        (grid_active_range[0][0], grid_active_range[1][0]+1),
        (grid_active_range[0][1], grid_active_range[1][1]+1),
        (grid_active_range[0][2], grid_active_range[1][2]+1),
    )

    for i, j in ti.ndrange(active[0], active[1]):
        grid_baseidx[i, j, active[2][0]] = grid_cnt[i, j, active[2][0]]
        for k in range(active[2][0]+1, active[2][1]):
            grid_baseidx[i, j, k] = grid_baseidx[i, j, k-1] + grid_cnt[i, j, k]

    for i in ti.ndrange(active[0]):
        grid_scnt[i, active[1][0]] = grid_baseidx[i, active[1][0], active[2][1]-1]
        for j in range(active[1][0]+1, active[1][1]):
            grid_scnt[i, j] = grid_scnt[i, j-1] + grid_baseidx[i, j, active[2][1]-1]

    grid_s2cnt[active[0][0]] = grid_scnt[active[0][0], active[1][1]-1]
    ti.loop_config(serialize=True)
    for i in range(active[0][0]+1, active[0][1]):
        grid_s2cnt[i] = grid_s2cnt[i-1] + grid_scnt[i, active[1][1]-1]

    for i, j, k in ti.ndrange(active[0], active[1], active[2]):
        if i>active[0][0]:
            grid_baseidx[i, j, k] += grid_s2cnt[i-1]
        if j>active[1][0]:
            grid_baseidx[i, j, k] += grid_scnt[i, j-1]
        grid_baseidx[i, j, k] -= grid_cnt[i, j, k]
        grid_cnt[i, j, k] = 0

    max_margin = 0.0

    for i in range(start, end):
        mid_v = (v[f[i][0]] + v[f[i][1]] + v[f[i][2]])/3
        ti.atomic_max(max_margin, (mid_v-v[f[i][0]]).norm())
        ti.atomic_max(max_margin, (mid_v-v[f[i][1]]).norm())
        ti.atomic_max(max_margin, (mid_v-v[f[i][2]]).norm())
        idx = grid_idx(mid_v)
        pid = grid_baseidx[idx] + ti.atomic_add(grid_cnt[idx], 1)
        particle_v[pid, 0] = v[f[i][0]]
        particle_v[pid, 1] = v[f[i][1]]
        particle_v[pid, 2] = v[f[i][2]]
        particle_idx[pid, 0] = f[i][0]
        particle_idx[pid, 1] = f[i][1]
        particle_idx[pid, 2] = f[i][2]
    
    detect_ub = grid_h-max_margin
    # if detect_ub<0.01:
    #     print("detect ub:", detect_ub)
        # exit(0)

@ti.kernel
def project_pair_self(sys:ti.template(), start: ti.i32, end: ti.i32, body_idx:ti.i32, debug:ti.i32):
    for i in range(start, end):
        xq = sys.pos[i]
        q_idx = grid_idx(xq)
        r0 = ti.max(q_idx-ti.Vector([1, 1, 1]), grid_active_range[0])
        r1 = ti.min(q_idx+ti.Vector([1, 1, 1]), grid_active_range[1]) + ti.Vector([1, 1, 1])
        d_min = 1e6
        cos_max = -1e6
        proj_flag = 0
        proj_idx = ti.Vector([0, 0, 0])
        proj_w = vec3(0)
        cnt = 0
        for gi, gj, gk in ti.ndrange((r0[0], r1[0]), (r0[1], r1[1]), (r0[2], r1[2])):

            for pid in range(grid_baseidx[gi, gj, gk], grid_baseidx[gi, gj, gk]+grid_cnt[gi, gj, gk]):

                if i == particle_idx[pid, 0] or i == particle_idx[pid, 1] or i == particle_idx[pid, 2]:
                    continue

                v1 = particle_v[pid, 0]
                v2 = particle_v[pid, 1]
                v3 = particle_v[pid, 2]

                c, d, w = pt2tri(xq, v1, v2, v3)
                if c != 0:
                    continue
                cnt += 1
                vt = v1*w[0] + v2*w[1] + v3*w[2]
                nt = (v2-v1).cross(v3-v1).normalized()
                cos = (xq-vt).dot(nt)
                if d < d_min-1e-5 or (d < d_min+1e-5 and cos>cos_max):
                    d_min = d
                    cos_max = cos
                    proj_idx[0] = particle_idx[pid, 0]
                    proj_idx[1] = particle_idx[pid, 1]
                    proj_idx[2] = particle_idx[pid, 2]
                    proj_w = w
                    proj_flag = 1

        if debug:
            print(d_min, cnt)

        v1 = sys.pos[proj_idx[0]]
        v2 = sys.pos[proj_idx[1]]
        v3 = sys.pos[proj_idx[2]]
        n1 = sys.vn[proj_idx[0]]
        n2 = sys.vn[proj_idx[1]]
        n3 = sys.vn[proj_idx[2]]
        v = proj_w[0]*v1 + proj_w[1]*v2 + proj_w[2]*v3
        n = proj_w[0]*n1 + proj_w[1]*n2 + proj_w[2]*n3
        if debug and proj_flag==1:
            nt = (v2 - v1).cross(v3 - v1).normalized()
            print(n, nt)
        temp_proj_dir = sys.proj_dir[body_idx, i]
        if sys.proj_flag[body_idx, i] == 0 and proj_flag == 1:
            sys.proj_dir[body_idx, i] = (xq-v).dot(n) > 0
        if sys.proj_flag[body_idx, i] and sys.proj_dir[body_idx, i] != temp_proj_dir:
            print("???")
        # if proj_flag == 1 and body_idx == 2 and sys.bel(start + 1) == 1:
        #     proj_num[None] += 1
        #     if proj_num[None] % 10 == 0:
        #         print(n.normalized(), sys.proj_dir[body_idx, i])
        sys.proj_flag[body_idx, i] = proj_flag
        sys.proj_idx[body_idx, i] = proj_idx
        sys.proj_w[body_idx, i] = proj_w

@ti.kernel
def project_pair(sys:ti.template(), start: ti.i32, end: ti.i32, body_idx:ti.i32, debug:ti.i32):
    for i in range(start, end):
        xq = sys.pos[i]
        q_idx = grid_idx(xq)
        r0 = ti.max(q_idx-ti.Vector([1, 1, 1]), grid_active_range[0])
        r1 = ti.min(q_idx+ti.Vector([1, 1, 1]), grid_active_range[1]) + ti.Vector([1, 1, 1])
        d_min = 1e6
        cos_max = -1e6
        proj_flag = 0
        proj_idx = ti.Vector([0, 0, 0])
        proj_w = vec3(0)
        for gi, gj, gk in ti.ndrange((r0[0], r1[0]), (r0[1], r1[1]), (r0[2], r1[2])):
            for pid in range(grid_baseidx[gi, gj, gk], grid_baseidx[gi, gj, gk]+grid_cnt[gi, gj, gk]):
                v1 = particle_v[pid, 0]
                v2 = particle_v[pid, 1]
                v3 = particle_v[pid, 2]

                c, d, w = pt2tri(xq, v1, v2, v3)
                vt = v1*w[0] + v2*w[1] + v3*w[2]
                nt = (v2-v1).cross(v3-v1).normalized()
                cos = (xq-vt).dot(nt)
                if d < d_min-1e-5 or (d < d_min+1e-5 and cos>cos_max):
                    d_min = d
                    cos_max = cos
                    proj_idx[0] = particle_idx[pid, 0]
                    proj_idx[1] = particle_idx[pid, 1]
                    proj_idx[2] = particle_idx[pid, 2]
                    proj_w = w
                    if c==0:
                        proj_flag = 1
                    elif c>0:
                        proj_flag = not sys.border_flag[particle_idx[pid, c-1]]
                    else:
                        p1 = particle_idx[pid, 2] if c!=-3 else particle_idx[pid, 0]
                        p2 = particle_idx[pid, 2+c] if c!=-3 else particle_idx[pid, 1]
                        proj_flag = not (sys.border_flag[p1] and sys.border_flag[p2])
        v1 = sys.pos[proj_idx[0]]
        v2 = sys.pos[proj_idx[1]]
        v3 = sys.pos[proj_idx[2]]
        n1 = sys.vn[proj_idx[0]]
        n2 = sys.vn[proj_idx[1]]
        n3 = sys.vn[proj_idx[2]]
        v = proj_w[0]*v1 + proj_w[1]*v2 + proj_w[2]*v3
        n = proj_w[0]*n1 + proj_w[1]*n2 + proj_w[2]*n3
        temp_proj_dir = sys.proj_dir[body_idx, i]
        if sys.proj_flag[body_idx, i] == 0 and proj_flag == 1:
            sys.proj_dir[body_idx, i] = (xq-v).dot(n) > 0
        if sys.proj_flag[body_idx, i] and sys.proj_dir[body_idx, i] != temp_proj_dir:
            print("???")
        # if proj_flag == 1 and body_idx == 2 and sys.bel(start + 1) == 1:
        #     proj_num[None] += 1
        #     if proj_num[None] % 10 == 0:
        #         print(n.normalized(), sys.proj_dir[body_idx, i])
        sys.proj_flag[body_idx, i] = proj_flag
        sys.proj_idx[body_idx, i] = proj_idx
        sys.proj_w[body_idx, i] = proj_w

def projection_query(sys, debug=False, self_contact=[]):
    # proj_num[None] = 0
    for body_idx, body in enumerate(sys.body_list):
        p2g(sys.pos, sys.faces, body.f_start, body.f_end)
        for body_idx2, body2 in enumerate(sys.body_list):
            if body_idx2 != body_idx: # TODO: contact relationship
                project_pair(sys, body2.v_start, body2.v_end, body_idx, debug)
        if body_idx in self_contact:
            project_pair_self(sys, body.v_start, body.v_end, body_idx, debug)

    # print("potential contact", proj_num[None])