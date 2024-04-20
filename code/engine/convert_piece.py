import numpy as np
import trimesh

from . import readfile

def comp(x, y):
    if x > y:
        return (y, x)
    else:
        return (x, y)

def get_normal(x, y, z):
    v1 = y - x
    v2 = z - x
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return normal

def get_bound(faces):
    edge_count = dict()             # For detecting boundary
    bounds = set()
    bound_points = set()
    
    for face in faces:
        d1, d2, d3 = face[0], face[1], face[2]
        e1, e2, e3 = (d1, d2), (d2, d3), (d3, d1)
        if e1 in edge_count or e2 in edge_count or e3 in edge_count:
            raise Exception("Same direction face")
        edge_count[e1] = True
        edge_count[e2] = True
        edge_count[e3] = True
    
    for e in edge_count:
        d1, d2 = e
        if (d2, d1) in edge_count:
            continue
        bounds.add(e)
        bound_points.add(d1)
        bound_points.add(d2)

    return bounds, bound_points

def get_flank(vertices, faces):
    nv = vertices.shape[0]
    flank_faces = list()
    bounds, _ = get_bound(faces)
    
    for e in bounds:
        d1, d2 = e
        nd1, nd2 = d1 + nv, d2 + nv
        flank_faces.append([d2, d1, nd1])
        flank_faces.append([nd1, nd2, d2])
    
    return np.array(flank_faces)

def print_mesh_info(mesh: trimesh.Trimesh):
    print('Num Vertices:', mesh.vertices.shape)
    print('Num Faces:', mesh.faces.shape)
    print('Num Vertex Normals:', mesh.vertex_normals.shape)
    print('Num Face Normals:', mesh.face_normals.shape)
    
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        print('Num Face Colors:', mesh.visual.face_colors.shape)
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        print('Num Vertex UV:', mesh.visual.uv.shape)
    
    print()

def thick_cloth(mesh, corners, thickness, both_sides):
    print("Begin thicken cloth...")
    new_mesh = mesh.copy()
    faces = mesh.faces
    vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals
    nv = vertices.shape[0]
    
    print_mesh_info(mesh)
    
    # the lower layer
    new_vertices = vertices - vertex_normals * thickness
    new_faces = faces[:, ::-1]
    new_mesh.vertices = new_vertices
    new_mesh.faces = new_faces
    new_vertex_normals = new_mesh.vertex_normals        # auto calculated
    
    flank_faces = get_flank(vertices, faces)
    new_faces = new_faces + nv
    corners = corners + [d + nv for d in corners]
    
    # visual
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        face_colors = mesh.visual.face_colors
        mesh.visual.face_colors = np.vstack([
            face_colors, face_colors, np.tile(face_colors[:1], (flank_faces.shape[0], 1))
        ])
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        if both_sides:
            u = mesh.visual.uv[:, :1]
            v = mesh.visual.uv[:, 1:]
            
            old_uv = np.concatenate([      u / 2.0, v], axis=1)
            new_uv = np.concatenate([1.0 - u / 2.0, v], axis=1)
            mesh.visual.uv = np.vstack([old_uv, new_uv])
        else:
            mesh.visual.uv = np.vstack([mesh.visual.uv, mesh.visual.uv])
    mesh.vertices = np.vstack([vertices, new_vertices])
    mesh.faces = np.vstack([faces, new_faces, flank_faces])
    mesh.vertex_normals = np.vstack([vertex_normals, new_vertex_normals])
    
    return mesh, corners


def subdivide_cloth(mesh, corners):
    print("Begin subdivide cloth...")
    faces = mesh.faces
    vertices, vertex_faces = mesh.vertices, mesh.vertex_faces
    edges, edges_face = mesh.edges, mesh.edges_face
    bounds, bound_points = get_bound(faces)
    nv = vertices.shape[0]
    nf = faces.shape[0]
    ne = (edges.shape[0] - len(bounds)) // 2 + len(bounds)

    print_mesh_info(mesh)
    
    new_vertices = np.zeros((nv, 3))
    new_edge_vertices = np.zeros((ne, 3))
    new_face_vertices = np.zeros((nf, 3))
    edge_midpoint = np.zeros((ne, 3))
    
    new_faces = list()
    unique_edge = dict()
    edge_index = dict()
    new_corners = list()
    
    # Build face points
    for i, face in enumerate(faces):
        d1, d2, d3 = face[0], face[1], face[2]
        new_face_vertices[i] = (vertices[d1] + vertices[d2] + vertices[d3]) / 3
    
    # Build edge points
    for i, edge in enumerate(edges):
        d1, d2 = edge[0], edge[1]
        
        if (d2, d1) in unique_edge:
            edge_index[(d1, d2)] = edge_index[(d2, d1)]
            unique_edge[(d2, d1)] = (i, unique_edge[(d2, d1)])
        else:
            edge_index[(d1, d2)] = len(unique_edge)
            unique_edge[(d1, d2)] = i
            ide = edge_index[(d1, d2)]
            edge_midpoint[ide] = (vertices[d1] + vertices[d2]) / 2
    
    for edge in unique_edge:
        d1, d2 = edge
        ide = edge_index[(d1, d2)]
        
        if (d1, d2) in bounds or (d1 in corners and d2 in corners):
            new_edge_vertices[ide] = edge_midpoint[ide]
            if d1 in corners and d2 in corners:
                new_corners.append(ide + nv)
        else:
            ide1, ide2 = unique_edge[(d1, d2)]
            face_mid = (new_face_vertices[edges_face[ide1]] +
                        new_face_vertices[edges_face[ide2]]) / 2
            new_edge_vertices[ide] = (edge_midpoint[ide] + face_mid) / 2

    # print(len(bounds))
    # print(len(unique_edge), ne)

    def add_face_vertex(d, el, er, idf):
        # el: (d, dl), er: (dr, d)
        if d in bound_points:
            if el in bounds:
                new_vertices[d] += vertices[el[1]]
            if er in bounds:
                new_vertices[d] += vertices[er[0]]
        else:
            idel, ider = edge_index[el], edge_index[er]
            new_vertices[d] += new_face_vertices[idf] + edge_midpoint[idel] + edge_midpoint[ider]

    # Build vertex points
    for i, face in enumerate(faces):
        d1, d2, d3 = face[0], face[1], face[2]
        e1, e2, e3 = (d1, d2), (d2, d3), (d3, d1)
        
        add_face_vertex(d1, e1, e3, i)
        add_face_vertex(d2, e2, e1, i)
        add_face_vertex(d3, e3, e2, i)
        
    for i in range(nv):
        if i in bound_points:
            new_vertices[i] = (new_vertices[i] + vertices[i] * 6) / 8
        else:
            lnf = (list(vertex_faces[i]) + [-1]).index(-1)
            new_vertices[i] = (new_vertices[i] / lnf + vertices[i] * (lnf - 3)) / lnf
    
    # Build Topology
    for i, face in enumerate(faces):
        d1, d2, d3 = face[0], face[1], face[2]
        e1, e2, e3 = (d1, d2), (d2, d3), (d3, d1)
        ide1, ide2, ide3 = edge_index[e1], edge_index[e2], edge_index[e3]
        
        idf = i + nv + ne
        ide1, ide2, ide3 = ide1 + nv, ide2 + nv, ide3 + nv
        
        new_faces.append([idf, d1, ide1])
        new_faces.append([idf, ide1, d2])
        new_faces.append([idf, d2, ide2])
        new_faces.append([idf, ide2, d3])
        new_faces.append([idf, d3, ide3])
        new_faces.append([idf, ide3, d1])
            
    new_faces = np.array(new_faces)
    
    # Visual
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        face_colors = mesh.visual.face_colors
        new_face_colors = list()
        for i, face in enumerate(faces):
            new_face_colors.append(np.tile(face_colors[i: i + 1], (6, 1)))
        new_face_colors = np.vstack(new_face_colors)
        mesh.visual.face_colors = new_face_colors
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        vertex_uv = mesh.visual.uv
        edge_uv = np.zeros((ne, 2))
        face_uv = np.zeros((nf, 2))

        for edge in unique_edge:
            d1, d2 = edge
            ide = edge_index[(d1, d2)]
            edge_uv[ide] = (vertex_uv[d1] + vertex_uv[d2]) / 2
            
        for i, face in enumerate(faces):
            d1, d2, d3 = face[0], face[1], face[2]
            face_uv[i] = (vertex_uv[d1] + vertex_uv[d2] + vertex_uv[d3]) / 3
            
        mesh.visual.uv = np.vstack([vertex_uv, edge_uv, face_uv])
        
    # Update mesh
    corners = corners + new_corners
    mesh.vertices = np.vstack([new_vertices, new_edge_vertices, new_face_vertices])
    mesh.faces = new_faces
    
    return mesh, corners

def process_cloth(mesh, corners, thickness, sub_times, both_sides):
    inter_res = list()
    inter_res.append([mesh, corners])

    mesh, corners = thick_cloth(inter_res[-1][0], inter_res[-1][1], thickness, both_sides)
    inter_res.append([mesh, corners])
    for i in range(sub_times):
        mesh, corners = subdivide_cloth(inter_res[-1][0], inter_res[-1][1])
        inter_res.append([mesh, corners])

    print_mesh_info(mesh)
    
    return mesh, corners, inter_res
    # thick_mesh.export(thick_obj)

def read_obj_faces(filename):
    faces = list()
    fin = open(filename, "r")
    for line in fin:
        if line.startswith("f"):
            items = line.split(" ")
            face = [int(items[i].split("/")[0]) for i in range(1, 4)]
            faces.append(face)
    
    return faces

if __name__ == "__main__":
    input_obj = "../imgs/small_cloth.obj"
    # input_obj = "../imgs/new_cloth.obj"
    
    thick_obj = "../imgs/small_thick_cloth.obj"
    # thick_obj = "../imgs/new_thick_cloth.obj"
    sub_obj = "../imgs/small_sub_cloth.obj"
    
    output_obj = "../imgs/small_thick_sub_cloth_2.obj"
    # output_obj = "../imgs/new_thick_sub_cloth.obj"
    
    # Load an .obj file
    # mesh = trimesh.load_mesh(input_obj, process=False)
    mesh = trimesh.load_mesh(input_obj, skip_material=True)
    print(type(mesh.visual))
    print(hasattr(mesh.visual, "uv"))
    corners = [0, 1, 2, 3]
    
    new_mesh, corners = process_cloth(mesh, corners, 0.1, 2)
    new_mesh.export(output_obj)
    # exit()
    # fin = open(input_obj, "r")
    # mesh = trimesh.exchange.obj.load_obj(fin, skip_material=True)
    # print(mesh)
    
    # manual_faces = np.array(read_obj_faces(input_obj)) - 1
    # print(np.concatenate([faces, manual_faces], axis=1))
    # print()
    # sub_mesh = subdivide_cloth(mesh, corners)
    # sub_mesh.export(sub_obj)

    
