import trimesh
import taichi as ti
from queue import Queue
from PIL import Image
from typing import List

from . import convert_piece
from .build_luisa_script import *
from assets_lookup import cube_corner_uvs

class ThinShellRenderOptions:
    def __init__(self):
        pass

    def get_luisa(self, attr_name):
        attr_value = getattr(self, attr_name)
        if attr_value is None:
            return None
        else:
            return attr_value.to_luisa()
        
class TextureOptions(ThinShellRenderOptions):
    def __init__(
        self,
        color         : tuple   = None,
        file          : str     = None,
        image         : Image   = None,
        image_scale   : float   = 1.0,
        checker_on    : "TextureOptions" = None,
        checker_off   : "TextureOptions" = None,
        checker_scale : float            = None,
        mix_method    : str              = "mix",
        mix_factor    : float            = 0.5,
        mix_top       : "TextureOptions" = None,
        mix_bottom    : "TextureOptions" = None,
        uv_remap      : "TextureOptions" = None,
        uv_texture    : "TextureOptions" = None,
    ):
        self.color = color
        self.file = file
        self.image = image
        self.image_scale = (image_scale,)
        self.checker_on = checker_on
        self.checker_off = checker_off
        self.checker_scale = checker_scale
        self.mix_method = mix_method    
        self.mix_factor = mix_factor    
        self.mix_top = mix_top       
        self.mix_bottom = mix_bottom
        self.uv_remap = uv_remap
        self.uv_texture = uv_texture

    def to_luisa(self):
        return LuisaTexture(
            constant=self.color,
            file=self.file,
            image=self.image,
            image_scale=self.image_scale,
            checker_on=self.get_luisa("checker_on"),
            checker_off=self.get_luisa("checker_off"),
            checker_scale=self.checker_scale,
            mix_method=self.mix_method,
            mix_factor=self.mix_factor,
            mix_top=self.get_luisa("mix_top"),
            mix_bottom=self.get_luisa("mix_bottom"),
            uv_remap=self.get_luisa("uv_remap"),
            uv_texture=self.get_luisa("uv_texture"),
        )

class CameraOptions(ThinShellRenderOptions):
    def __init__(
        self,
        position,
        look_at,
        up=(0.0, 0.0, 1.0),
        resolution=(1024, 1024)
    ):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.resolution = resolution
    
    def to_luisa(self):
        return LuisaCamera(
            position=self.position,
            look_at=self.look_at,
            up=self.up,
            resolution=self.resolution
        )

class LightOptions(ThinShellRenderOptions):
    def __init__(self, position, color):
        self.position = position
        self.color = color

class EnvironmentOptions(ThinShellRenderOptions):
    def __init__(
        self,
        texture  : TextureOptions = TextureOptions(color=(1.0, 1.0, 1.0)),
        rotation : float = 0,
    ):
        self.texture = texture
        self.rotation = rotation

class BodyOptions(ThinShellRenderOptions):
    def __init__(
        self,
        texture: TextureOptions,
        roughness: TextureOptions,
        normal: TextureOptions,
        eta: TextureOptions,
        opacity: TextureOptions,
        clamp_normal: float,
    ):
        self.texture = texture
        self.roughness = roughness
        self.normal = normal
        self.eta = eta
        self.opacity = opacity
        self.clamp_normal = clamp_normal
    
    def to_luisa(self, target_opacity: float=None):
        return LuisaSurface(
            material="plastic",
            roughness=self.get_luisa("roughness"),
            normal=self.get_luisa("normal"),
            opacity=self.get_luisa("opacity") if target_opacity is None else \
                LuisaTexture(constant=(target_opacity,)),
            kd=self.get_luisa("texture"),
            eta=self.get_luisa("eta"),
        )

class ClothOptions(BodyOptions):
    def __init__(
        self,
        texture=None,
        roughness=None,
        normal=None,
        eta=TextureOptions(color=(1.0,)),
        opacity=TextureOptions(color=(1.0,)),
        clamp_normal=-1,
        both_sides=False,
        target=None,
        freeze_frame=None,
        app_vel=None,
        curve=False,
        thickness=None,
    ):
        if roughness is None:
            roughness = TextureOptions(color=(0.8,))
        BodyOptions.__init__(self, texture, roughness, normal, eta, opacity, clamp_normal)
        self.both_sides = both_sides
        self.target = target
        self.freeze_frame = freeze_frame
        self.app_vel = app_vel
        self.curve = curve
        if thickness is None:
            self.thickness = 0.0002
        else:
            self.thickness = thickness

class ElasticOptions(BodyOptions):
    def __init__(
        self,
        texture=None,
        roughness=None,
        normal=None,
        eta=TextureOptions(color=(1.3,)),
        opacity=TextureOptions(color=(1.0,)),
        clamp_normal=-1,
        target=None,
        lower=None,
        app_vel=None,
        rigid=False,
    ):
        if roughness is None:
            roughness = TextureOptions(color=(0.5,))
        BodyOptions.__init__(self, texture, roughness, normal, eta, opacity, clamp_normal)
        self.target = target
        self.lower = lower
        self.app_vel = app_vel
        self.rigid = rigid

class GroundOptions(BodyOptions):
    def __init__(
        self,
        texture=TextureOptions(
            checker_on  = TextureOptions(color=(0.8, 0.8, 0.8)),
            checker_off = TextureOptions(color=(0.3, 0.5, 0.7)),
            checker_scale = 300
        ),
        roughness=TextureOptions(color=(0.1,)),
        normal=None,
        eta=TextureOptions(color=(3.0,)),
        opacity=TextureOptions(color=(1.0,)),
        clamp_normal=-1,
        height: float = 0,
        range: float = 30
    ):
        BodyOptions.__init__(self, texture, roughness, normal, eta, opacity, clamp_normal)
        self.height = height
        self.range = range

class TableOptions(BodyOptions):
    def __init__(
        self,
        texture=None,
        roughness=TextureOptions(color=(0.1,)),
        normal=None,
        eta=TextureOptions(color=(1.5,)),
        opacity=TextureOptions(color=(1.0,)),
        clamp_normal=-1,
        file: str = None,
        up_limit: float = 0,
        right_limit: float = None,
        rotation: float = 0,
        scale: float = 0,
        replace_first: bool = False,
    ):
        BodyOptions.__init__(self, texture, roughness, normal, eta, opacity, clamp_normal)
        self.file = file
        self.up_limit = up_limit
        self.right_limit = right_limit
        self.rotation = rotation
        self.scale = scale
        self.replace_first = replace_first

def get_cube_uvs(n1, n2, left_bottom, right_up):
    def lin(f, t, l, r):
        return l + (r - l) * f / t
    
    uvs = list()
    for index in range(n1 * n2):
        x1 = index % n1
        x2 = index // n1
        uvs.append([
            lin(x1, n1 - 1, left_bottom[0], right_up[0]),
            lin(x2, n2 - 1, left_bottom[1], right_up[1])
        ])
    return np.array(uvs)

def build_elastic_mesh(elastic, offset):
    vertices = elastic.F_x.to_numpy(dtype=float) + offset
    # if lower is not None:
    #     vertices[:, 2] = vertices[:, 2] - lower
    faces = elastic.f2v.to_numpy(dtype=int)

    if not hasattr(elastic, "n_cube") or not hasattr(elastic, "load") or elastic.load:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if hasattr(elastic, "uv"):
            uvs = elastic.uv.to_numpy(dtype=float)
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)
        return mesh

    # up, down, front, back, left, right
    cube_dicts = [dict() for _ in range(6)]
    cube_faces = [list() for _ in range(6)]
    cube_vertices = [list() for _ in range(6)]
    cube_uvs = [list() for _ in range(6)]
    # print("Cube size:", elastic.n_cube, "vertices:", vertices.shape, elastic.n_verts)

    cube_ranges = [
        (elastic.n_cube[1], elastic.n_cube[0]),
        (elastic.n_cube[1], elastic.n_cube[0]),
        (elastic.n_cube[2], elastic.n_cube[1]),
        (elastic.n_cube[2], elastic.n_cube[1]),
        (elastic.n_cube[2], elastic.n_cube[0]),
        (elastic.n_cube[2], elastic.n_cube[0])
    ]
    cube_bounds = [
        (2, elastic.n_cube[2] - 1), (2, 0),
        (0, elastic.n_cube[0] - 1), (0, 0),
        (1, elastic.n_cube[1] - 1), (1, 0)
    ]
    for ig in np.ndindex(*elastic.n_cube):
        p = (ig[0] * elastic.n_cube[1] + ig[1]) * elastic.n_cube[2] + ig[2]
        for i in range(6):
            if ig[cube_bounds[i][0]] == cube_bounds[i][1]:
                cube_dicts[i][p] = len(cube_dicts[i])
                cube_vertices[i].append(vertices[p])

    for face in faces:
        for i in range(6):
            cube_dict = cube_dicts[i]
            if face[0] in cube_dict and face[1] in cube_dict and face[2] in cube_dict:
                cube_faces[i].append([cube_dict[face[0]], cube_dict[face[1]], cube_dict[face[2]]])
                break

    tot_verts = 0
    for i in range(6):
        cur_vertices = np.array(cube_vertices[i])
        cur_faces = np.array(cube_faces[i])
        cur_uvs = get_cube_uvs(
            cube_ranges[i][0], cube_ranges[i][1],
            cube_corner_uvs[i][0], cube_corner_uvs[i][1]
        )

        cube_vertices[i] = cur_vertices
        cube_faces[i] = cur_faces + tot_verts
        cube_uvs[i] = cur_uvs
        tot_verts += cur_vertices.shape[0]
    
    vertices = np.concatenate(cube_vertices, axis=0)
    faces = np.concatenate(cube_faces, axis=0)
    uvs = np.concatenate(cube_uvs, axis=0)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # print(vertices.shape[0], mesh.vertices.shape[0])
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)
    return mesh

def get_mix_texture(n, m, curve_judge, division=4):
    img_size = (1024, 1024, 3)       # height, width
    print(f"cloth n={n}, m={m}, img_size={img_size}")
    img = np.zeros(img_size, dtype=float)
    img_queue = Queue()
    for i in range((n + 1) * division):
        for j in range((m + 1) * division):
            if i % division == 0:
                pij = (i // division) * (m + 1) + (j // division)
                curve_name = curve_judge(pij)
                cur_color = (1, 0, 0) if curve_name == "down" else \
                            (0, 0, 1) if curve_name == "up" else (1, 1, 1)
            else:
                cur_color = (1, 1, 1)
            # puv = uvs[pij]
            ci = img_size[0] - 1 - min(int(i / (n * division) * img_size[0]), img_size[0] - 1)
            cj = img_size[1] - 1 - min(int(j / (m * division) * img_size[1]), img_size[1] - 1)
            img[ci, cj] = cur_color
            # print((i, j), (ci, cj))
            img_queue.put((ci, cj))
            
    d = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    while not img_queue.empty():
        ci, cj = img_queue.get()
        for i in range(4):
            nci = ci + d[i][0]
            ncj = cj + d[i][1]
            if min(nci, ncj) < 0 or nci >= img_size[0] or ncj >= img_size[1]:
                continue
            if img[nci, ncj, 0] == 0 and img[nci, ncj, 1] == 0 and img[nci, ncj, 2] == 0:
                img[nci, ncj] = img[ci, cj]
                img_queue.put((nci, ncj))

    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img

def build_cloth_mesh(cloth, both_sides, thickness, offset):
    vertices = cloth.pos.to_numpy(dtype=float) + offset
    faces = cloth.f2v.to_numpy(dtype=int)
    uvs = cloth.uv.to_numpy(dtype=float)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs)
    
    n = cloth.N
    m = cloth.M if hasattr(cloth, "M") else n
        
    sub_times = 2
    corners = [0, m, n * (m + 1), (n + 1) * (m + 1) - 1]
    new_mesh, new_corners, inter_result = convert_piece.process_cloth(
        mesh, corners,
        thickness=thickness,
        sub_times=sub_times,
        both_sides=both_sides
    )
    
    return new_mesh

def get_mesh(
    body_mesh: trimesh.Trimesh,
    mat_name: str = None,
    clamp_normal: float = -1,
):
    uvs = body_mesh.visual.uv if hasattr(body_mesh.visual, "uv") else None
    return LuisaMesh(
        vertices=body_mesh.vertices,
        triangles=body_mesh.faces,
        normals=body_mesh.vertex_normals,
        uvs=uvs,
        surface=mat_name,
        clamp_normal=clamp_normal,
    )

def process_curve_mix(sys, cloth_textures : List[ClothOptions]):
    for i, cloth_texture in enumerate(cloth_textures):
        cloth = sys.cloths[i]
        if cloth_texture.curve:
            curve_judge = \
                lambda x: "up" if sys.is_upper_curve_py(x) else \
                          "down" if sys.is_lower_curve_py(x) else None
            
            n = cloth.N
            m = cloth.M if hasattr(cloth, "M") else n
            cloth_mix = get_mix_texture(n, m, curve_judge)
            cloth_textures[i] = TextureOptions(
                mix_top=cloth_texture.texture,
                mix_bottom=TextureOptions(image=cloth_mix),
                mix_factor=1.0,
                mix_method="multiply"
            )

def build_global_scene(
    frame_scripts : LuisaRenderScripts,
    camera_option : CameraOptions,
    cloth_textures : List[ClothOptions],
    elastic_textures : List[ElasticOptions],
    light_options : List[LightOptions] = [],
    environment_option : EnvironmentOptions = None,
    table_option : TableOptions = None,
    ground_option : GroundOptions = None,
):
    # Add camera
    frame_scripts.add_shared_camera(name="view_port", camera=camera_option.to_luisa())

    # Add lights
    for i, light_option in enumerate(light_options):
        simple_light(
            index=i,
            position=light_option.position,
            color=light_option.color,
            scripts=frame_scripts,
        )

    # Add environment
    if environment_option is not None:
        simple_background(
            environment_option.texture.to_luisa(),
            axis=(0, 0, 1),
            rotation=environment_option.rotation,
            scripts=frame_scripts,
        )

    # Add table
    if table_option is not None:
        frame_scripts.add_shared_surface(
            name="mat_table",
            surface=table_option.to_luisa(),
        )
        simple_table(
            file=table_option.file,
            up_limit=table_option.up_limit,
            right_limit=table_option.right_limit,
            rotation=table_option.rotation,
            scale=table_option.scale,
            surface="mat_table",
            clamp_normal=table_option.clamp_normal,
            scripts=frame_scripts,
        )
        elastic_start = 1 if table_option.replace_first else 0
    else:
        elastic_start = 0

    # Add ground
    if ground_option is not None:
        frame_scripts.add_shared_surface(
            name="mat_ground",
            surface=ground_option.to_luisa(),
        )
        simple_ground(
            height=ground_option.height,
            range=ground_option.range,
            surface="mat_ground",
            scripts=frame_scripts,
        )

    # Add cloth texture
    for i, cloth_texture in enumerate(cloth_textures):
        frame_scripts.add_shared_surface(
            name=f"mat_cloth_{i}",
            surface=cloth_texture.to_luisa(),
        )
    
    # Add elastic texture
    for i, elastic_texture in enumerate(elastic_textures):
        si = i + elastic_start
        frame_scripts.add_shared_surface(
            name=f"mat_elastic_{si}",
            surface=elastic_texture.to_luisa()
        )

def build_taichi_scene(
    sys,
    frame: str,
    frame_scripts: LuisaRenderScripts,
    cloth_textures: List[ClothOptions],
    elastic_textures: List[ElasticOptions],
    replace_first: bool=False,
    camera_option: CameraOptions=None,
    preview: bool=True,
):
    n_cloth = len(sys.cloths)
    n_elastic = len(sys.elastics)
    elastic_start = 1 if replace_first else 0 
    if len(cloth_textures) != n_cloth or len(elastic_textures) + elastic_start != n_elastic:
        raise Exception("Texture numbers and entity numbers do not match!")
    
    script = frame_scripts.get_script(frame, create=True)

    if camera_option is not None:
        script.add_camera(name="view_port", camera=camera_option.to_luisa())

    for i, cloth_texture in enumerate(cloth_textures):
        cloth_offset = np.array([0.0, 0.0, 0.0])
        cloth_mesh = build_cloth_mesh(
            sys.cloths[i], cloth_texture.both_sides,
            cloth_texture.thickness, cloth_offset,
        )
        script.add_mesh(
            name=f"cloth_{i}",
            mesh=get_mesh(
                body_mesh=cloth_mesh,
                mat_name=f"mat_cloth_{i}",
                clamp_normal=cloth_texture.clamp_normal,
            ),
        )

    # Load robot
    # if robot_folder is not None:
    #     frame_int = int(frame)
    #     if frame_int > 0 and frame_int < tot_timestep:
    #         vert_file = f"vert_{frame_int - 1}.npy"
    #         face_file = f"faces_{frame_int - 1}.npy"
    #     elif frame_int >= tot_timestep:
    #         vert_file = f"vert_{tot_timestep - 2}.npy"
    #         face_file = f"faces_{tot_timestep - 2}.npy"
    #     elif frame_int < 0:
    #         vert_file = f"prev_vert_{frame_int + prev_timestep}.npy"
    #         face_file = f"prev_faces_{frame_int + prev_timestep}.npy"
    #     else:
    #         raise Exception("Render frame 0!")
    #     robot_real_offset = np.array([0.0, 0.0, 0.0]) if robot_offset is None else np.array(robot_offset)
    #     if robot_app_vel is not None:
    #         robot_real_offset += np.array(robot_app_vel) * app_step
    #     robot_vertices_file = os.path.join(robot_folder, vert_file)
    #     robot_faces_file = os.path.join(robot_folder, face_file)
    #     robot_vertices = np.load(robot_vertices_file) + robot_real_offset
    #     robot_faces = np.load(robot_faces_file).reshape(-1, 3)
    #     robot_mesh = trimesh.Trimesh(
    #         vertices = robot_vertices,
    #         faces = robot_faces,
    #     )
    #     # robot_mesh.export(os.path.join(robot_folder, f"robot_mesh_{frame}.obj"))
    #     script.add_mesh(
    #         name=f"dexterous",
    #         mesh=get_mesh(
    #             body_mesh=robot_mesh,
    #             mat_name="mat_dexterous"
    #         )
    #     )

    for i, elastic_texture in enumerate(elastic_textures):
        si = i + elastic_start
        elastic_offset = np.array([0.0, 0.0, 0.0])
        elastic_obj = sys.elastics[si]
        elastic_mesh = build_elastic_mesh(elastic_obj, elastic_offset)

        script.add_mesh(
            name=f"elastic_{si}",
            mesh=get_mesh(
                body_mesh=elastic_mesh,
                mat_name=f"mat_elastic_{si}",
                clamp_normal=elastic_texture.clamp_normal,
            )
        )
    
    if preview:
        frame_scripts.export_scripts()
