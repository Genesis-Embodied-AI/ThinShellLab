import os
import shutil
import numpy as np
import trimesh
import math
import copy
from PIL import Image

def to_rad(deg: float):
    return deg / 180.0 * math.pi

def get_tabs(tab: int):
    return "".join(["\t" for _ in range(tab)])

def get_str_list(l: tuple):
    return [str(c) for c in l]

def get_list(l: tuple):
    return ", ".join(get_str_list(l))

def get_vert_list(l: list, tab: int):
    tabs = get_tabs(tab)
    return ",\n".join([f"{tabs}{c}" for c in l])

def get_matrix(m: np.ndarray, tab: int):
    if m.shape != (4, 4):
        raise Exception(f"Invalid matrix shape: {m.shape}")
    tabs = get_tabs(tab)
    return ",\n".join([f"{tabs}{get_list(m[i])}" for i in range(4)])

def get_properties(l: list):
    return "\n".join(l)

def check_plugin(plugin_type, plugin_name, plugin_list):
    if plugin_name not in plugin_list:
        raise Exception(f"Invalid {plugin_type}: {plugin_name}! Requires: {plugin_list} ")

def get_light_color(color: tuple, intensity: float):
    return tuple([c * intensity for c in color])

def matrix_from_quat(quat: tuple):      # Here quat is xyzw format
    new_quat = (quat[3], quat[0], quat[1], quat[2])     # to wxyz format
    return trimesh.transformations.quaternion_matrix(new_quat)

def export_without_mtl(mesh, path: str):
    obj_str = trimesh.exchange.obj.export_obj(
        mesh, include_texture=True, write_texture=False
    )
    obj_list = obj_str.split("\n")
    obj_list = [s for s in obj_list if not s.startswith("usemtl") and 
                                       not s.startswith("mtllib") and 
                                       not s.startswith("#")]
    obj_str = "\n".join(obj_list)
    fw = open(path, "w")
    fw.write(obj_str)
    fw.close()

class LuisaOption:
    def __init__(self):
        pass

    def check_attr(self, attr_list):
        for attr in attr_list:
            if getattr(self, attr) is None:
                return False
        return True
    
    def make_shared(self, scripts: "LuisaRenderScripts"):
        shared_option = copy.deepcopy(self)
        for attr in dir(shared_option):
            if not attr.startswith('__'):
                value = getattr(shared_option, attr)
                if not callable(value) and isinstance(value, LuisaOption):
                    new_value = value.make_shared(scripts)
                    setattr(shared_option, attr, new_value)
        return shared_option

class LuisaTexture(LuisaOption):
    def __init__(
        self,
        constant      : tuple          = None,     # constant
        file          : str            = None,     # image
        file_shared   : bool           = False,
        image         : Image          = None,
        image_scale   : tuple          = None,
        checker_on    : "LuisaTexture" = None,     # checkerboard
        checker_off   : "LuisaTexture" = None,
        checker_scale : float          = None,
        mix_method    : str            = None,     # mix
        mix_factor    : float          = None,
        mix_top       : "LuisaTexture" = None,
        mix_bottom    : "LuisaTexture" = None,
        uv_remap      : "LuisaTexture" = None,     # uv
        uv_texture    : "LuisaTexture" = None,
    ):
        if mix_method is not None:
            check_plugin("mix_method", mix_method, ["add", "substract", "multiply", "mix"])
        self.constant = constant
        self.file = file
        self.file_shared = file_shared
        self.image = image
        self.image_scale = image_scale
        self.checker_on = checker_on
        self.checker_off = checker_off
        self.checker_scale = checker_scale
        self.mix_method = mix_method    
        self.mix_factor = mix_factor    
        self.mix_top = mix_top       
        self.mix_bottom = mix_bottom
        self.uv_remap = uv_remap
        self.uv_texture = uv_texture

    def export_script(self, script: "LuisaRenderScript", tab=0):
        tabs = get_tabs(tab)
        if self.check_attr(["constant"]):
            return f'''constant {{
{tabs}\tv {{ {get_list(self.constant)} }}
{tabs}}}'''
        
        elif self.check_attr(["file"]) or self.check_attr(["image"]):
            if self.check_attr(["file"]):
                image_name = script.add_global_image(self.file, self.file_shared)
            else:
                image_name = script.add_global_image()
                image_file = os.path.join(script.script_dir, image_name)
                self.image.save(image_file)
            return f'''image {{
{tabs}\tfile {{ \"{image_name}\" }}
{tabs}\tscale {{ {get_list(self.image_scale)} }}
{tabs}}}'''
        
        elif self.check_attr(["checker_on", "checker_off", "checker_scale"]):
            return f'''checkerboard {{
{tabs}\ton: {self.checker_on.export_script(script, tab + 1)}
{tabs}\toff: {self.checker_off.export_script(script, tab + 1)}
{tabs}\tscale {{ {self.checker_scale} }}
{tabs}}}'''
        
        elif self.check_attr(["mix_top", "mix_bottom", "mix_method", "mix_factor"]):
            return f'''mix {{
{tabs}\ttop: {self.mix_top.export_script(script, tab + 1)}
{tabs}\tbottom: {self.mix_bottom.export_script(script, tab + 1)}
{tabs}\tfactor {{ {self.mix_factor} }}
{tabs}\tmethod {{ \"{self.mix_method}\" }}
{tabs}}}'''
        
        elif self.check_attr(["uv_remap", "uv_texture"]):
            return f'''uvmapping {{
{tabs}\tuv_map: {self.uv_remap.export_script(script, tab + 1)}
{tabs}\ttexture: {self.uv_texture.export_script(script, tab + 1)}
{tabs}}}'''
        
        else:
            raise Exception("Invalid texture type!")

    def make_shared(self, scripts: "LuisaRenderScripts"):
        shared_option = super().make_shared(scripts)    
        if shared_option.check_attr(["image"]):
            shared_option.file = scripts.add_global_image()
            shared_option.file_shared = True
            image_file = os.path.join(scripts.script_dir, shared_option.file)
            shared_option.image.save(image_file)
            shared_option.image = None
        elif shared_option.check_attr(["file"]):
            shared_option.file = scripts.add_global_obj(shared_option.file)
            shared_option.file_shared = True
        return shared_option
    
class LuisaTransform(LuisaOption):
    def __init__(
        self,
        matrix        : np.ndarray = None,
        srt_scale     : tuple      = None,
        srt_rotate    : tuple      = None,
        srt_translate : tuple      = None,
    ):
        self.matrix = matrix
        self.srt_scale = srt_scale
        self.srt_rotate = srt_rotate
        self.srt_translate = srt_translate
    
    def export_script(self, script: "LuisaRenderScript", tab=0):
        tabs = get_tabs(tab)
        if self.check_attr(["matrix"]):
            return f'''matrix {{
{tabs}\tm {{
{get_matrix(self.matrix, tab + 2)}
{tabs}\t}}
{tabs}}}'''
        elif self.check_attr(["srt_scale", "srt_rotate", "srt_translate"]):
            return f'''srt {{
{tabs}\tscale {{ {get_list(self.srt_scale)} }}
{tabs}\trotate {{ {get_list(self.srt_rotate)} }}
{tabs}\ttranslate {{ {get_list(self.srt_translate)} }}
{tabs}}}'''
        else:
            raise Exception("Invalid transform type!")

class LuisaLight(LuisaOption):
    def __init__(
        self,
        emission : LuisaTexture = LuisaTexture(constant=(1.0, 1.0, 1.0))
    ):
        self.emission = emission

    def export_script(self, script: "LuisaRenderScript", tab=0):
        tabs = get_tabs(tab)
        return f'''diffuse {{
{tabs}\temission: {self.emission.export_script(script, tab + 1)}
{tabs}}}'''

class LuisaSurface(LuisaOption):
    def __init__(
        self,
        material  : str          = "plastic",
        roughness : LuisaTexture = LuisaTexture(constant=(0.0,)),
        normal    : LuisaTexture = None,
        opacity   : LuisaTexture = None,
        kd        : LuisaTexture = None,
        ks        : LuisaTexture = None,
        kt        : LuisaTexture = None,
        eta       : LuisaTexture = LuisaTexture(constant=(1.5,)),
        eta_name  : str          = "Al"
    ):
        check_plugin("material", material, ["plastic", "glass", "metal"])
        self.material = material
        if roughness is None:
            self.roughness = LuisaTexture(constant=(0.0,))
        else:
            self.roughness = roughness
        self.normal = normal
        self.opacity = opacity
        self.kd = kd
        self.ks = ks
        self.kt = kt
        self.eta = eta
        self.eta_name = eta_name

    def export_script(self, script: "LuisaRenderScript", tab=0):
        properties_content = list()
        tabs = get_tabs(tab)
        properties_content.append(f"{tabs}\troughness: {self.roughness.export_script(script, tab + 1)}")
        if self.normal is not None:
            properties_content.append(f"{tabs}\tnormal_map: {self.normal.export_script(script, tab + 1)}")
        if self.opacity is not None:
            properties_content.append(f"{tabs}\topacity: {self.opacity.export_script(script, tab + 1)}")
        if (self.material == "plastic" or self.material == "metal") and self.kd is not None:
            properties_content.append(f"{tabs}\tKd: {self.kd.export_script(script, tab + 1)}")
        if (self.material == "plastic" or self.material == "glass") and self.ks is not None:
            properties_content.append(f"{tabs}\tKs: {self.ks.export_script(script, tab + 1)}")
        if (self.material == "glass") and self.kt is not None:
            properties_content.append(f"{tabs}\tKt: {self.kt.export_script(script, tab + 1)}")
        if self.material == "plastic" or self.material == "glass":
            properties_content.append(f"{tabs}\teta: {self.eta.export_script(script, tab + 1)}")
        elif self.material == "metal":
            properties_content.append(f"{tabs}\teta {{ \"{self.eta_name}\" }}")
        return f'''{self.material} {{
{get_properties(properties_content)}
{tabs}}}'''

class LuisaMesh(LuisaOption):
    def __init__(
        self,
        file         : str            = None,
        file_shared  : bool           = False,
        vertices     : np.ndarray     = None,
        triangles    : np.ndarray     = None,
        normals      : np.ndarray     = None,
        uvs          : np.ndarray     = None,
        plane_div    : int            = None,
        sphere_div   : int            = None,
        transform    : LuisaTransform = None,
        surface      : str            = None,
        light        : LuisaLight     = None,
        clamp_normal : float          = None,
    ):
        self.file = file
        self.file_shared = file_shared
        self.vertices = vertices
        self.triangles = triangles
        self.normals = normals
        self.uvs = uvs
        self.plane_div = plane_div
        self.sphere_div = sphere_div
        self.transform = transform
        self.surface = surface
        self.light = light
        self.clamp_normal = clamp_normal

    def build_mesh(self):
        mesh = trimesh.Trimesh(
            vertices       = self.vertices,
            faces          = self.triangles,
            vertex_normals = self.normals
        )
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=self.uvs)
        return mesh

    def export_script(self, script: "LuisaRenderScript", tab=0):
        tabs = get_tabs(tab)
        properties_content = list()
        if self.check_attr(["file"]) or self.check_attr(["vertices", "triangles"]):
            if self.check_attr(["file"]):
                obj_name = script.add_global_obj(self.file, self.file_shared)
            else:
                obj_name = script.add_global_obj()
                obj_file = os.path.join(script.script_dir, obj_name)
                export_without_mtl(self.build_mesh(), obj_file)
            mesh_name = "mesh"
            properties_content.append(f"{tabs}\tfile {{ \"{obj_name}\" }}")
        elif self.check_attr(["plane_div"]):
            mesh_name = "plane"
            properties_content.append(f"{tabs}\tsubdivision {{ {self.plane_div} }}")
        elif self.check_attr(["sphere_div"]):
            mesh_name = "sphere"
            properties_content.append(f"{tabs}\tsubdivision {{ {self.sphere_div} }}")
        else:
            raise Exception("Invalid mesh type!")
        
        if self.transform is not None:
            properties_content.append(f"{tabs}\ttransform: {self.transform.export_script(script, tab + 1)}")
        if self.surface is not None:
            if self.surface not in script.surfaces:
                raise Exception(f"Surface {self.surface} not added!")
            properties_content.append(f"{tabs}\tsurface {{ @{self.surface} }}")
        if self.light is not None:
            properties_content.append(f"{tabs}\tlight: {self.light.export_script(script, tab + 1)}")
        if self.clamp_normal is not None:
            properties_content.append(f"{tabs}\tclamp_normal {{ {self.clamp_normal} }}")
        return f'''{mesh_name} {{
{get_properties(properties_content)}
{tabs}}}'''

    def make_shared(self, scripts: "LuisaRenderScripts"):
        shared_option = super().make_shared(scripts)
        if shared_option.check_attr(["vertices", "triangles"]):
            shared_option.file = scripts.add_global_obj()
            shared_option.file_shared = True
            obj_file = os.path.join(scripts.script_dir, shared_option.file)
            export_without_mtl(shared_option.build_mesh(), obj_file)
            shared_option.vertices, shared_option.triangles = None, None
            shared_option.normals, shared_option.uvs = None, None
        elif shared_option.check_attr(["file"]):
            shared_option.file = scripts.add_global_obj(shared_option.file)
            shared_option.file_shared = True
        return shared_option

class LuisaCamera(LuisaOption):
    def __init__(
        self,
        position   : tuple,
        look_at    : tuple,
        up         : tuple = (0, 0, 1),
        fov        : float = 30,
        spp        : int   = 256,
        resolution : tuple = (600, 600)
    ):
        self.position = position   
        self.look_at = look_at    
        self.up = up         
        self.fov = fov        
        self.spp = spp        
        self.resolution = resolution 

    def export_script(self, script: "LuisaRenderScript", tab=0):
        tabs = get_tabs(tab)
        return f'''pinhole {{
{tabs}\tposition {{ {get_list(self.position)} }}
{tabs}\tlook_at {{ {get_list(self.look_at)} }}
{tabs}\tup {{ {get_list(self.up)} }}
{tabs}\tfov {{ {self.fov} }}
{tabs}\tspp {{ {self.spp} }}
{tabs}\tfilter: gaussian {{
{tabs}\t\tradius {{ 1 }}
{tabs}\t}}
{tabs}\tfilm: color {{
{tabs}\t\tresolution {{ {get_list(self.resolution)} }}
{tabs}\t}}
{tabs}}}'''

class LuisaEnvironment(LuisaOption):
    def __init__(
        self,
        emission: LuisaTexture,
        transform: LuisaTransform = None,
    ):
        self.emission = emission
        self.transform = transform

    def export_script(self, script: "LuisaRenderScript", tab=0):
        tabs = get_tabs(tab)
        return f'''spherical {{
{tabs}\temission: {self.emission.export_script(script, tab + 1)}
{tabs}\ttransform: {self.transform.export_script(script, tab + 1)}
{tabs}}}'''

class LuisaRenderScript:
    def __init__(
        self,
        script_dir   : str,
        mark         : str   = None,
        integrator   : str   = "wavepath_v2",
        sampler      : str   = "pmj02bn",
        spectrum     : str   = "hero",
        clamp_normal : float = -1,
    ):
        check_plugin("integrator", integrator, ["wavepath", "wavepath_v2"])
        check_plugin("sampler", sampler, ["independent", "pmj02bn"])
        check_plugin("spectrum", spectrum, ["hero", "srgb"])
        
        self.script_dir = script_dir
        if mark is None:
            self.script_name = "scene.luisa" 
            self.model_name = "models"
            self.texture_name = "textures"
        else:
            self.script_name = f"scene_{mark}.luisa"
            self.model_name = f"models_{mark}"
            self.texture_name = f"textures_{mark}"
        self.integrator = integrator
        self.sampler = sampler
        self.spectrum = spectrum
        self.clamp_normal = clamp_normal

        self.environment = None
        self.surfaces = dict()
        self.meshes = dict()
        self.cameras = dict()
        self.light_count = 0

        self.images = dict()
        self.image_objects = dict()
        self.objs = dict()
    
    def add_environment(self, environment: LuisaEnvironment, replace=True):
        if self.environment is None or replace:
            self.environment = environment
    
    def add_surface(self, name: str, surface: LuisaSurface, replace=True):
        if name not in self.surfaces or replace:
            self.surfaces[name] = surface
    
    def add_mesh(self, name: str, mesh: LuisaMesh, replace=True):
        if name not in self.meshes or replace:
            self.meshes[name] = mesh

    def add_camera(self, name: str, camera: LuisaCamera, replace=True):
        if name not in self.cameras or replace:
            self.cameras[name] = camera

    # def add_global_image(self, image_path: str=None, image_object: Image=None, shared: bool=False):
    def add_global_image(self, path: str=None, shared: bool=False):
        # if path is not None:
        if path in self.images:
            return self.images[path]
        if shared:
            assert path is not None, "Empty shared image."
            self.images[path] = path
            return path
        index = len(self.images)
        ext = os.path.splitext(path)[1]
        new_path = os.path.join(self.texture_name, f"image_{index}{ext}")
        if path is None:
            path = new_path
        self.images[path] = new_path
        return new_path
    
    def add_global_obj(self, path: str=None, shared: bool=False):
        if path in self.objs:
            return self.objs[path]
        if shared:
            assert path is not None, "Empty shared object."
            self.objs[path] = path
            return path
        index = len(self.objs)
        new_path = f"{self.model_name}/obj_{index}.obj"
        if path is None:
            path = new_path
        self.objs[path] = new_path
        return new_path
    
    def export_script(self, rebuild=True):
        if rebuild and os.path.exists(self.script_dir):
            shutil.rmtree(self.script_dir)
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(os.path.join(self.script_dir, self.model_name), exist_ok=True)
        os.makedirs(os.path.join(self.script_dir, self.texture_name), exist_ok=True)
        export_file = os.path.join(self.script_dir, self.script_name)
        self.objs.clear()
        self.images.clear()
        fout = open(export_file, "w")
        print(f"Export to file: {export_file}")

        # Add surfaces
        for name in self.surfaces:
            surface = self.surfaces[name]
            fout.write(f"surface {name}: {surface.export_script(self, 0)}\n\n")

        # Add shapes
        for name in self.meshes:
            mesh = self.meshes[name]
            fout.write(f"shape {name}: {mesh.export_script(self, 0)}\n\n")
            
        # Add cameras
        for name in self.cameras:
            camera = self.cameras[name]
            fout.write(f"camera {name}: {camera.export_script(self, 0)}\n\n")

        # Build script root
        camera_refs = [f"@{name}" for name in self.cameras]
        shape_refs = [f"@{name}" for name in self.meshes]

        # Add environment
        properties_content = list()
        properties_content.append(f"\tclamp_normal {{ {self.clamp_normal} }}")
        properties_content.append(f'''\tintegrator: {self.integrator} {{
\t\tsampler: {self.sampler} {{}}
\t}}''')
        properties_content.append(f"\tspectrum: {self.spectrum} {{}}")
        properties_content.append(f'''\tcameras {{
{get_vert_list(camera_refs, 2)}
\t}}''')
        properties_content.append(f'''\tshapes {{
{get_vert_list(shape_refs, 2)}
\t}}''')
        if self.environment is not None:
            properties_content.append(f"\tenvironment: {self.environment.export_script(self, 1)}")
        fout.write(f'''render {{
{get_properties(properties_content)}
}}''')
        fout.close()

        for image_path in self.images:
            if image_path != self.images[image_path]:
                new_path = os.path.join(self.script_dir, self.images[image_path])
                shutil.copyfile(image_path, new_path)
        for obj_path in self.objs:
            if obj_path != self.objs[obj_path]:
                new_path = os.path.join(self.script_dir, self.objs[obj_path])
                shutil.copyfile(obj_path, new_path)

        print(f"Export finished.")

class LuisaRenderScripts:
    def __init__(
        self,
        script_dir   : str,
        # marks        : list  = [],
        integrator   : str   = "wavepath_v2",
        sampler      : str   = "pmj02bn",
        spectrum     : str   = "hero",
        clamp_normal : float = -1
    ):
        '''
            script_dir/textures_shared/     texture folder
            script_dir/models_shared/       model folder
            script_dir/textures_mark/       texture folder
            script_dir/models_mark/         model folder
            script_dir/scene_mark.luisa     scene script
        '''
        self.script_dir = script_dir
        self.scripts = dict()
        self.integrator = integrator
        self.sampler = sampler
        self.spectrum = spectrum
        self.clamp_normal = clamp_normal
        # if len(marks) == 0:
        #     raise Exception("No scripts!")
        # self.scripts = {
        #     mark: LuisaRenderScript(
        #         script_dir, mark, integrator, sampler, spectrum, clamp_normal
        #     ) for mark in marks
        # }

        self.environment = None
        self.meshes = dict()
        self.surfaces = dict()
        self.cameras = dict()

        self.images = dict()
        self.objs = dict()
    
    def get_script(self, mark, create=False):
        if mark not in self.scripts:
            if create:
                self.scripts[mark] = LuisaRenderScript(
                    self.script_dir, mark,
                    self.integrator, self.sampler, self.spectrum, self.clamp_normal
                )
            else:
                raise Exception(f"No script named {mark}.")
        return self.scripts[mark]

    def add_shared_environment(self, environment: LuisaEnvironment):
        self.environment = environment

    def add_shared_surface(self, name: str, surface: LuisaSurface):
        self.surfaces[name] = surface

    def add_shared_mesh(self, name: str, mesh: LuisaMesh):
        self.meshes[name] = mesh
    
    def add_shared_camera(self, name: str, camera: LuisaCamera):
        self.cameras[name] = camera

    def add_global_image(self, path: str=None):
        if path in self.images:
            return self.images[path]
        index = len(self.images)
        ext = os.path.splitext(path)[1]
        new_path = os.path.join("textures_shared", f"image_{index}{ext}")
        if path is None:
            path = new_path
        self.images[path] = new_path
        return new_path
            
    def add_global_obj(self, path: str=None):
        if path in self.objs:
            return self.objs[path]
        index = len(self.objs)
        new_path = os.path.join("models_shared", f"obj_{index}.obj")
        if path is None:
            path = new_path
        self.objs[path] = new_path
        return new_path

    def export_scripts(self, update_mark=False):
        if os.path.exists(self.script_dir):
            shutil.rmtree(self.script_dir)
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(os.path.join(self.script_dir, "models_shared"), exist_ok=True)
        os.makedirs(os.path.join(self.script_dir, "textures_shared"), exist_ok=True)
        self.objs.clear()
        self.images.clear()

        # Add environment
        if self.environment is not None:
            shared_environment = self.environment.make_shared(self)
            for script in self.scripts.values():
                script.add_environment(shared_environment, False)

        # Add surfaces
        for name, surface in self.surfaces.items():
            shared_surface = surface.make_shared(self)
            for script in self.scripts.values():
                script.add_surface(name, shared_surface, False)

        # Add shapes
        for name, mesh in self.meshes.items():
            shared_mesh = mesh.make_shared(self)
            for script in self.scripts.values():
                script.add_mesh(name, shared_mesh, False)

        # Add cameras
        for name, camera in self.cameras.items():
            shared_camera = camera.make_shared(self)
            for script in self.scripts.values():
                script.add_camera(name, shared_camera, False)
        
        for image_path in self.images:
            if image_path != self.images[image_path]:
                new_path = os.path.join(self.script_dir, self.images[image_path])
                shutil.copyfile(image_path, new_path)
        for obj_path in self.objs:
            if obj_path != self.objs[obj_path]:
                new_path = os.path.join(self.script_dir, self.objs[obj_path])
                shutil.copyfile(obj_path, new_path)
        
        for script in self.scripts.values():
            script.export_script(rebuild=False)

        if update_mark:
            with open(os.path.join(self.script_dir, "update_mark.txt"), "w") as update_f:
                print("Generate update mark!")

def simple_background(
    texture: LuisaTexture, axis: tuple, rotation: float,
    script: LuisaRenderScript=None, scripts: LuisaRenderScripts=None,
):
    background_env = LuisaEnvironment(
        emission=texture,
        transform=LuisaTransform(
            srt_scale=(1.0, 1.0, 1.0),
            srt_rotate=(axis[0], axis[1], axis[2], rotation),
            srt_translate=(0.0, 0.0, 0.0),
        )
    )
    if scripts is not None:
        scripts.add_shared_environment(background_env)
    elif script is not None:
        script.add_environment(background_env)
    else:
        raise Exception("No scripts for background!")

def simple_ground(
    height: float, range: float, surface: str,
    script: LuisaRenderScript = None, scripts: LuisaRenderScripts = None,
):
    ground_shape = LuisaMesh(
        plane_div=3,
        transform=LuisaTransform(
            srt_scale=(range, range, 1.0),
            srt_rotate=(0.0, 0.0, 1.0, 0.0),
            srt_translate=(0.0, 0.0, height)
        ),
        surface=surface,
    )
    if scripts is not None:
        scripts.add_shared_mesh("ground", ground_shape)
    elif script is not None:
        script.add_mesh("ground", ground_shape)
    else:
        raise Exception("No scripts for ground!")

def simple_table(
    file: str, up_limit: float, right_limit: float,
    rotation: float, scale: float, surface: str, clamp_normal: float,
    script: LuisaRenderScript = None, scripts: LuisaRenderScripts = None,
):
    table_mesh = trimesh.load_mesh(file, skip_material=True)
    table_vertices = table_mesh.vertices * scale
    table_zmax = np.max(table_vertices[:, 2])

    if right_limit is None:
        table_translate = (0, 0, up_limit - table_zmax)
    else:
        table_dis = right_limit - np.max(table_vertices[:, 0])
        table_translate = (table_dis * math.cos(to_rad(rotation)),
                            table_dis * math.sin(to_rad(rotation)),
                            up_limit - table_zmax)

    table_shape = LuisaMesh(
        file=file,
        transform=LuisaTransform(
            srt_scale=(scale, scale, scale),
            srt_rotate=(0.0, 0.0, 1.0, rotation),
            srt_translate=table_translate
        ),
        surface=surface,
        clamp_normal=clamp_normal,
    )
    if scripts is not None:
        scripts.add_shared_mesh("table", table_shape)
    elif script is not None:
        script.add_mesh("table", table_shape)
    else:
        raise Exception("No scripts for table!")

def simple_light(
    index: int, position: tuple, color: tuple,
    radius: float = 0.01, intensity: float = 20.0,
    script: LuisaRenderScript = None, scripts: LuisaRenderScripts = None,
):
    light_shape = LuisaMesh(
        sphere_div=3,
        transform=LuisaTransform(
            srt_scale=(radius, radius, radius),
            srt_rotate=(0.0, 0.0, 1.0, 0.0),
            srt_translate=position
        ),
        light=LuisaLight(
            emission=LuisaTexture(constant=get_light_color(color, intensity))
        )
    )
    light_name = f"light_{index}"
    if scripts is not None:
        scripts.add_shared_mesh(light_name, light_shape)
    elif script is not None:
        script.add_mesh(light_name, light_shape)
    else:
        raise Exception("No scripts for light!")
