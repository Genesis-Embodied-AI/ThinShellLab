### Path
- In your repository folder, add these folders to your `~/.bashrc`:
```
export PYTHONPATH=$PYTHONPATH:${PWD}/code/engine
export PYTHONPATH=$PYTHONPATH:${PWD}/code/task_scene
export PYTHONPATH=$PYTHONPATH:${PWD}/code/training
```

### Render
- Here are two ways to render our scene, Taichi GGUI and LuisaRender Script. Taichi GGUI renders real-time image in GUI windows with low resolution, and LuisaRender Script generates meta-data script files for high-resolution and more realistic rendering outputs. This can be specified using the option `--render_option`.
- To run LuisaRender Script, necessary assets should be loaded. Run `git submodule update --init --recursive` to load the submodule `AssetLoader` and run `export PYTHONPATH=$PYTHONPATH:${PWD}/data/AssetLoader` to add the asset path to `PYTHONPATH`.
- For seeing the rendering results of LuisaRender Script, you should setup LuisaRender and use the command `` to get the outputs.