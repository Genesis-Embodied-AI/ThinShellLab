def read_node(filename="../data/tactile.node"):

    file = open(filename, encoding='utf-8')
    contents2 = file.readline()
    array = contents2.split(' ')
    tot_cnt = int(array[0])
    # print(contents2)
    # # 利用循环全部读出
    positions = []
    maxx = 0.0
    for i in range(tot_cnt):
        contents2 = file.readline()
        array = [float(i) for i in contents2.split(" ") if i.strip()]
        positions.append(array[1:])
        maxx = max(array[3], maxx)
    file.close()
    # print(maxx)
    # print("tot nodes: ", tot_cnt)
    return tot_cnt, positions

def read_smesh(filename="../data/tactile.face"):

    file = open(filename, encoding='utf-8')
    contents2 = file.readline()
    # print(contents2)
    array = contents2.split(' ')
    tot_cnt = int(array[0])

    f2v = []

    for i in range(tot_cnt):
        contents2 = file.readline()
        array = [int(i) for i in contents2.split(" ") if i.strip()]
        f2v.append(array[1:4])
    file.close()

    return tot_cnt, f2v

def read_ele(filename="../data/tactile.ele"):

    file = open(filename, encoding='utf-8')
    contents2 = file.readline()
    # print(contents2)
    array = contents2.split(' ')
    tot_cnt = int(array[0])

    F_verts = []

    for i in range(tot_cnt):
        contents2 = file.readline()
        array = [int(i) for i in contents2.split(" ") if i.strip()]
        F_verts.append(array[1:])
    file.close()

    return tot_cnt, F_verts

def build_tactile_mesh():
    import trimesh
    filename = "../data/tactile3.obj"
    n_verts, nodes = read_node()
    n_surfaces, faces = read_smesh()
    mesh = trimesh.Trimesh(vertices=nodes, faces=faces)
    mesh.export(filename)
        
def save_ply():
    import open3d as o3d
    import numpy as np
    geometry = o3d.geometry.TriangleMesh()
    _, faces = read_smesh()
    geometry.triangles = o3d.utility.Vector3iVector(np.array(faces))
    _, verts = read_node()

    geometry.vertices = o3d.utility.Vector3dVector(np.array(verts))
    geometry.compute_vertex_normals()

    o3d.io.write_triangle_mesh(f"../surf_mesh.ply", geometry)

def read_hdf5(filename):
    import h5py
    f = h5py.File(filename, "r")
    return f

def read_force(filename):
    import numpy as np
    data = read_hdf5(filename)
    # for key in data.keys():
    #     print(key)
    force_data = np.array(data["force_measure"])
    # print(force_data)
    # force_data = force_data[::-1]
    # print("Force shape:", force_data.shape)
    force_data[:, 2] -= 0.44
    # print(force_data[450:550])
    return force_data

def read_pos(filename):
    import numpy as np
    data = read_hdf5(filename)
    for key in data.keys():
        print(key)
    pos_data = np.array(data["tool_pose"])
    print(pos_data[0])
    # print(pos_data[140:190])
    # print(pos_data[450:550])
    # print(pos_data[:, 2].min())
    return pos_data

def get_surface_new():
    import open3d as o3d
    import numpy as np
    mesh = o3d.io.read_triangle_mesh("../tactile.ply")
    vert = mesh.vertices
    vert = np.array(vert)
    for i in range(vert.shape[0]):
        if np.linalg.norm(vert[i]) < 0.007:
            vert[i] *= 0.0075 / 0.006
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    o3d.io.write_triangle_mesh(f"../tactile_new.ply", mesh)


def save_cloth_mesh(cloth, path):
    import open3d as o3d
    import numpy as np
    mesh = o3d.geometry.TriangleMesh()
    vertices = cloth.pos.to_numpy(dtype='float64')
    faces = cloth.f2v.to_numpy(dtype='float64')

    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(path, mesh)

def get_score(path, step, cmaes=False):
    import numpy as np
    rewards = np.load(path)[:step]
    if cmaes:
        rewards = -rewards - 5.1
    return rewards.max()

# get_surface_new()

# read_pos("../data/gelsight-force-capture-press-twist-x_2023-06-27-16-48-28.hdf5")
# read_force("../data/gelsight-force-capture-press-sphere_2023-06-16-12-38-26.hdf5")