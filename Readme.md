# ThinShellLab: Thin-Shell Object Manipulations With Differentiable Physics Simulations

---
<div align="center">
  <img src="images/teaser.gif"/>
</div> 

<p align="left">
    <a href='https://arxiv.org/abs/2404.00451'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://vis-www.cs.umass.edu/ThinShellLab/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

Official repo for the paper:

> **[Thin-Shell Object Manipulations With Differentiable Physics Simulations](https://vis-www.cs.umass.edu/ThinShellLab/)**  

ThinShellLab is a **fully differentiable** simulation platform tailored for **robotic interactions** with diverse thin-shell materials.

## Installation
You can create a Conda environment for this simulator first:
```bash
conda create -n thinshelllab python=3.9.16
conda activate thinshelllab
```

And install the package with its dependencies using
```bash
git clone https://github.com/wangyian-me/thinshelllab.git
cd thinshelllab
pip install -e .
```

### Render
- Here are two ways to render our scene, Taichi GGUI and LuisaRender Script. Taichi GGUI renders real-time image in GUI windows with low resolution, and LuisaRender Script generates meta-data script files for high-resolution and more realistic rendering outputs. This can be specified using the option `--render_option`.
- To run LuisaRender Script, necessary assets should be loaded. Run `git submodule update --init --recursive` to load the submodule `AssetLoader` and run `export PYTHONPATH=$PYTHONPATH:${PWD}/data/AssetLoader` to add the asset path to `PYTHONPATH`.
- For seeing the rendering results of LuisaRender Script, you should setup LuisaRender and use the command `` to get the outputs.

## Usage example

We put running scripts under code/scripts, you can simply run
```bash
cd thinshelllab
cd code
sh scripts/run_trajopt_folding.sh
```
to train a trajectory optimization policy for the folding task, or use other scripts to train on different tasks.

## Citation
If you find this codebase/paper useful for your research, please consider citing:
```
@inproceedings{wang2023thin,
  title={Thin-Shell Object Manipulations With Differentiable Physics Simulations},
  author={Wang, Yian and Zheng, Juntian and Chen, Zhehuan and Xian, Zhou and Zhang, Gu and Liu, Chao and Gan, Chuang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

