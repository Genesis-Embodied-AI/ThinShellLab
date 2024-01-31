import taichi as ti


import time
import numpy as np
import torch
import imageio
import sys, os
import json
loader_dir = "../ext/AssetLoader"
sys.path.insert(0, os.path.join(os.getcwd(), loader_dir))

import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from assets_lookup import texture_lookup, model_lookup, background_lookup
from convert_luisa import *

# from geometry import projection_query
from agent.traj_opt_single import agent_trajopt
import importlib

env_dict = {
    "balancing": "",
    "bouncing": "",
    "card": "",
    "folding": "",
    "forming": "",
    "interact": "",
    "lifting": "",
    "pick": "",
    "sliding": "",
}

def get_asset_maps(option, texture_scale, roughness):
    texture_map = TextureOptions(
        image=os.path.join(loader_dir, option.texture), image_scale=texture_scale
    )
    roughness_map = roughness if roughness is not None else \
        None if option.roughness is None else \
        TextureOptions(image=os.path.join(loader_dir, option.roughness))
    normal_map = None if option.normal is None else \
        TextureOptions(image=os.path.join(loader_dir, option.normal))
    return texture_map, roughness_map, normal_map

def get_asset_texture(texture_name, texture_scale=1, roughness=None):
    option = texture_lookup[texture_name]
    texture_map, roughness_map, normal_map = get_asset_maps(option, texture_scale, roughness)
    return texture_map, roughness_map, normal_map

def get_asset_model(model_name, texture_scale=1, roughness=None):
    option = model_lookup[model_name]
    texture_map, roughness_map, normal_map = get_asset_maps(option, texture_scale, roughness)
    model_file = os.path.join(loader_dir, option.object)
    return texture_map, roughness_map, normal_map, model_file

def get_asset_cloth(
    texture_name, image_scale=1, roughness=None,
    both_sides=False, curve=False, # thickness=None,
):
    texture_map, roughness_map, normal_map = get_asset_texture(texture_name, image_scale, roughness)
    return ClothOptions(
        texture=texture_map, roughness=roughness_map, normal=normal_map,
        both_sides=both_sides, curve=curve, # thickness=thickness
    )

def get_asset_elastic(texture_name, image_scale=1, roughness=None):
    texture_map, roughness_map, normal_map = get_asset_texture(texture_name, image_scale, roughness)
    return ElasticOptions(
        texture=texture_map, roughness=roughness_map, normal=normal_map,
    )
    
def get_asset_table(model_name, image_scale=1, roughness=None, clamp_normal=-1):
    texture_map, roughness_map, normal_map, model_file = get_asset_model(model_name, image_scale, roughness)
    return TableOptions(
        texture=texture_map, roughness=roughness_map, normal=normal_map,
        clamp_normal=clamp_normal, file=model_file,
    )

def get_asset_background(background_name, image_scale=1):
    texture_map = TextureOptions(
        image=os.path.join(loader_dir, background_lookup[background_name].texture),
        image_scale=image_scale,
    ),
    return EnvironmentOptions(texture=texture_map)
        
    # texture=TextureOptions(
    #     image=os.path.join(loader_dir, background_lookup["lebombo"].texture),
    #     image_scale=0.8,
    # ),

cloth_presets = {
    "cloth_1": get_asset_cloth("fabric_pattern_05"),
    "cloth_2": get_asset_cloth("fabric_pattern_07"),
    "genesis_paper": get_asset_cloth("genesis_logo"),
    "genesis_paper_curve": get_asset_cloth("genesis_logo", curve=True),
    "poker_1": get_asset_cloth("poker_1", both_sides=True, thickness=0.0005),
    "poker_2": get_asset_cloth("poker_2", both_sides=True, thickness=0.0005),
    "poker_3": get_asset_cloth("poker_3", both_sides=True, thickness=0.0005),
    "postcard_1": get_asset_cloth("postcard_1", both_sides=True, thickness=0.0005),
    "postcard_2": get_asset_cloth("postcard_2", both_sides=True, thickness=0.0005),
    "paper_1": ClothOptions(
        texture=TextureOptions(color=(0.9, 0.9, 0.9)),
        roughness=TextureOptions(color=(0.9,)),
        eta=TextureOptions(color=(1.3,)),
    ),
    "iron_1": ClothOptions(
        texture=TextureOptions(color=(0.6, 0.6, 0.6)),
        roughness=TextureOptions(color=(0.1,)),
        eta=TextureOptions(color=(20.0,)),
    ),
}
elastic_presets = {
    "wood_1": get_asset_elastic("dark_wood"),
    "wood_2": get_asset_elastic("laminate_floor"),
    "wood_3": get_asset_elastic("panel_wood"),
    "eraser": get_asset_elastic("eraser"),
    "paperbox": get_asset_elastic("paperbox", roughness=TextureOptions(color=(0.9,))),
    "pure_1": ElasticOptions(
        texture=TextureOptions(color=(1, 0.334, 0.52)),
        clamp_normal=0.95,
    ),
    "pure_2": ElasticOptions(
        texture=TextureOptions(color=(0.22, 0.72, 0.52)),
        clamp_normal=0.95,
    ),
    "pure_3": ElasticOptions(
        texture=TextureOptions(color=(0.09, 0.63, 0.90)),
        clamp_normal=0.95,
    ),
}
table_presets = {
    "wood_table_1": get_asset_table("wooden_table"),
    "wood_table_2": get_asset_table("wooden_plane", roughness=TextureOptions(color=(0.8,)), image_scale=1.5),
    "coffee_table": get_asset_table("coffee_table"),
    "round_table": get_asset_table("round_table", clamp_normal=0.999),
}
env_presets = {
    "indoor_1": get_asset_background("lebombo", image_scale=0.8),                       # rotation=155
    "indoor_2": get_asset_background("brown_photostudio_02", image_scale=0.9),          # rotation=144, 54
    "indoor_2_dark": get_asset_background("brown_photostudio_02", image_scale=0.7),     # rotation=144, -36
}

def parse_setting(preset_setting, setting_type):
    if setting_type == "cloth":
        cloth_preset = cloth_presets[preset_setting["type"]]
        if "thickness" in preset_setting:
            cloth_preset.thickness = float(preset_setting["thickness"])
        # if render_gif and "app_vel" in preset_setting:
        #     cloth_preset.app_vel = tuple(preset_setting["app_vel"])
        return cloth_preset

    elif setting_type == "elastic":
        elastic_preset = elastic_presets[preset_setting["type"]]
        # if "lower" in preset_setting:
        #     elastic_preset.lower = float(preset_setting["lower"])
        # if not render_gif and "target" in preset_setting:
        #     elastic_preset.target = preset_setting["target"]
        # if render_gif and "shared_target" in preset_setting:
        #     shared_target = preset_setting["shared_target"]
        #     elastic_preset.target = { shared_target: [
        #         [str(i) for i in range(1, int(shared_target))],
        #         preset_setting["target"][shared_target][1]
        #     ]}
        # if render_gif and "app_vel" in preset_setting:
        #     elastic_preset.app_vel = tuple(preset_setting["app_vel"])
        # if "rigid" in preset_setting:
        #     elastic_preset.rigid = preset_setting["rigid"]
        return elastic_preset
    
    elif setting_type == "table":
        table_preset = table_presets[preset_setting["type"]]
        table_preset.up_limit = float(preset_setting["up_limit"])
        if "right_limit" in preset_setting:
            table_preset.right_limit = float(preset_setting["right_limit"])
        table_preset.rotation = float(preset_setting["rotation"])
        table_preset.scale = float(preset_setting["scale"])
        table_preset.replace_first = bool(preset_setting["replace"])
        return table_preset
    
    elif setting_type == "environment":
        env_preset = env_presets[preset_setting["type"]]
        env_preset.rotation = float(preset_setting["rotation"])
        return env_preset

    elif setting_type == "camera":
        camera_preset = CameraOptions(position=(-0.15, 0.15, 0.04), look_at=(0, 0, 0))
        if preset_setting is not None:
            camera_preset.position = tuple(preset_setting["position"])
            camera_preset.look_at = tuple(preset_setting["look_at"])
            if "resolution" in preset_setting:
                camera_preset.resolution = tuple(preset_setting["resolution"])
        return camera_preset
    
    else:
        raise Exception("Invalid setting type.")
    
class TaichiRender:
    def __init__(self, scene_sys, env_name, save_dir=None, res=(800, 800), show_window=False,
                 background_color=(0.5, 0.5, 0.5), cam_pos=(-0.2, 0.2, 0.05), cam_lookat=(0, 0, 0)):
        self.scene_sys = scene_sys
        self.env_name = env_name
        self.window = ti.ui.Window(env_name, res=res, vsync=True, show_window=show_window)
        self.show_window = show_window
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color(background_color)
        self.scene = ti.ui.Scene()
        self.scene.ambient_light([0.8, 0.8, 0.8])
        self.scene.point_light((2, 2, 2), (1, 1, 1))

        self.camera = ti.ui.Camera()
        self.camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        self.camera.lookat(cam_lookat[0], cam_lookat[1], cam_lookat[2])
        self.camera.up(0, 0, 1)
        self.scene.set_camera(self.camera)

        self.vertex_colors = ti.Vector.field(3, dtype=float, shape=self.scene_sys.tot_NV)
        self.scene_sys.get_colors(self.vertex_colors)
    
    def render(self, frame: str):
        self.scene.mesh(
            self.scene_sys.x32, indices=self.scene_sys.f_vis,
            per_vertex_color=self.vertex_colors
        )
        self.canvas.scene(self.scene)
        if self.save_dir is not None:
            self.window.save_image(os.path.join(self.save_dir, f'{frame}.png'))
        if self.show_window:
            self.window.show()

class LuisaScriptRender:
    def __init__(self, scene_sys, env_name, save_dir, config_path="../data/scene_texture_options.json"):
        self.scene_sys = scene_sys
        self.env_name = env_name
        self.save_dir = save_dir

        render_dict = json.load(open(config_path, "r"))
        if self.env_name not in render_dict:
            raise Exception("Invalid environment name.")
        self.render_setting = render_dict[self.env_name]
        self.parse_options(self.render_setting)

        self.scripts = LuisaRenderScripts(
            script_dir=self.save_dir,
            integrator="wavepath_v2",
            sampler="pmj02bn",
            spectrum="hero",
            clamp_normal=-1,
        )
        
        process_curve_mix(self.scene_sys, self.cloth_options)
        build_global_scene(
            self.scripts, self.camera_option, self.cloth_options, self.elastic_options,
            light_options=self.light_options,
            environment_option=self.environment_option,
            table_option=self.table_option,
        )

    def parse_options(self, render_setting):
        self.environment_option = parse_setting(render_setting["environment"], "environment")
        self.light_options = [LightOptions(position=(2, 2, 2), color=(1, 1, 1))]
        self.camera_option = parse_setting(render_setting["camera"] if "camera" in render_setting else None, "camera")
        self.cloth_options = [parse_setting(cloth_setting, "cloth") for cloth_setting in render_setting["clothes"]]
        self.elastic_options = [parse_setting(elastic_setting, "elastic") for elastic_setting in render_setting["elastic"]]
        self.table_option = parse_setting(render_setting["table"], "table") if "table" in render_setting else None
        self.replace_first = self.table_option is not None and self.table_option.replace_first

    def render(self, frame: str, script_camera: CameraOptions=None, preview: bool=False):
        build_taichi_scene(
            self.scene_sys, frame, self.scripts,
            self.cloth_options, self.elastic_options, self.replace_first,
            camera_option=script_camera,
            preview=preview
        )

    def end_rendering(self):
        self.scripts.export_scripts()

class Renderer:
    def __init__(self, scene_sys, env_name, save_dir, option="Taichi", config_path=None):
        if option not in ["Taichi", "LuisaScript"]:
            raise Exception("Invalid renderer.")
        self.option = option
        if self.option == "Taichi":
            if config_path is None:
                self.renderer = TaichiRender(scene_sys, env_name, save_dir)
            else:
                taichi_dict = json.load(open(config_path, "r"))
                self.renderer = TaichiRender(
                    scene_sys, env_name, save_dir,
                    res=tuple(taichi_dict["resolution"]),
                    show_window=bool(taichi_dict["show_window"]),
                    background_color=tuple(taichi_dict["background_color"]),
                    cam_pos=tuple(taichi_dict["camera_pos"]),
                    cam_lookat=tuple(taichi_dict["camera_lookat"]),
                )
        else:
            if config_path is None:
                self.renderer = LuisaScriptRender(scene_sys, env_name, save_dir)
            else:
                self.renderer = LuisaScriptRender(scene_sys, env_name, save_dir, config_path)

    def render(self, frame: str, script_camera: CameraOptions=None, preview: bool=False):
        if self.option == "Taichi":
            self.renderer.render(frame)
        else:
            self.renderer.render(frame, script_camera, preview)

    def end_rendering(self):
        if self.option == "LuisaScript":
            self.renderer.end_rendering()