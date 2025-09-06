<div align=center>
  
# [ICCV 2025] *CHORDS*: Diffusion Sampling Accelerator with Multi-core Hierarchical ODE Solvers

Jiaqi Han*, Haotian Ye*, Puheng Li, Minkai Xu, James Zou, Stefano Ermon

**Stanford University**

<p>
<a href='https://arxiv.org/abs/2507.15260'><img src='https://img.shields.io/static/v1?&logo=arxiv&label=Paper&message=Arxiv:CHORDS&color=B31B1B'></a>
<a href='https://hanjq17.github.io/CHORDS/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
</p>

</div>


## 🎯 Introduction

We develop CHORDS, a general training-free and model-agnostic diffusion sampling acceleration strategy via multi-core parallelism. CHORDS views diffusion sampling as multi-core ODE solver pipeline, where slower yet accurate solvers progressively rectify faster solvers through a theoretically justified inter-core communication mechanism.

CHORDS significantly accelerates sampling across diverse large-scale image and video diffusion models, yielding up to 2.1x speedup with four cores, improving by 50% over baselines, and 2.9x speedup with eight cores, all without quality degradation.

![Overview](assets/chords-video.gif "Overview")

## 🛠 Dependencies

Our code relies on the following core packages:
```
torch
transformers
diffusers
hydra-core
imageio
imageio-ffmpeg
```
For the specific versions of these packages that have been verified as well as some optional dependencies, please refer to `requirements.txt`. We recommend creating a new virual environment via the following procedure:
```bash
conda create -n chords python=3.10
conda activate chords
pip install -r requirements.txt
```

## 🚀 Running Inference

Prior to running inference pipeline, please make sure that the models have been downloaded from 🤗 huggingface. We provide the download script for some example models for both image and video generation in `download.py`.


We use hydra to organize different hyperparameters for the image/video diffusion model as well as the sampling algorithm. The default configurations can be found under `configs` folder. The entries to launch the sampling for image and video generation are `run_image_mp.py` and `run_video_mp.py`, respectively.


### Text-to-Image (T2I)

The command below is an example to perform image generation using Flux with our algorithm CHORDS on 8 GPUs.

```bash
python run_image_mp.py \
    model=flux \
    ngpu=8 \
    output_base_path=output_samples_flux \
    prompt_file=prompts/image_demo.txt \
    algo=chords \
    algo.num_cores=8 \
    algo.init_t=0-2-4-8-16-24-32-40 \
    algo.stopping_kwargs.criterion=core_index \
    algo.stopping_kwargs.index=2
```

For `model` we currently support:
- `flux`: [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- `sd3-5`: [Stable Diffusion 3.5-Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)

 `ngpu` corresponds to the number of GPUs to use in parallel.
 
 `output_base_path` is the directory to save the generated samples.
 
 `prompt_file` stores the list of prompts, each per line, that will be sequentially employed to generate each image.

`algo.init_t` refers to the initialization sequence of CHORDS, as we have elaborated in the paper.

`algo.num_cores` refers to the number of cores, where we currently designate one core as one GPU since it already almost reaches the compute bound, therefore it should equal to `ngpu`.

`algorithm.stopping_kwargs` defines the early-stopping criteria, where `algo.stopping_kwargs.criterion` can be 
- `core_index`, which forces to return the output produced by the `algo.stopping_kwargs.index`-th fastest core. We recommend using 1 or 2.
- `residual`, which adaptively returns the output when the residual falls below certain `algo.stopping_kwargs.threshold`

For full functionality of the script, please refer to the arguments and their default values (such as the number of inference steps, the resolution of the image, etc.) under the `configs` folder, which will be automatically leveraged by hydra.

We also provide the script for the single-core sequential sampling baseline to facilitate comparison as follows:

```bash
python run_image_mp.py \
    model=flux \
    ngpu=1 \
    output_base_path=output_samples_flux \
    prompt_file=prompts/image_demo.txt \
    algo=sequential
```

### Text-to-Video (T2V)

Similarly, the following script can be used for video generation with CHORDS:

```bash
python run_video_mp.py \
    model=hunyuan \
    ngpu=8 \
    output_base_path=output_samples_hunyuan \
    prompt_file=prompts/video_demo.txt \
    algo=chords \
    algo.num_cores=8 \
    algo.init_t=0-2-4-8-16-24-32-40 \
    algo.stopping_kwargs.criterion=core_index \
    algo.stopping_kwargs.index=1
```
where for `model` we currently support:
- `hunyuan` [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)
- `wan2-1` [Wan2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
- `cogvideo` [CogVideo1.5X-5B](https://huggingface.co/zai-org/CogVideoX1.5-5B)

We recommend using 1 or 2 for `algo.stopping_kwargs.index`. If the video quality does not meet expectation, you can try increasing `algo.stopping_kwargs.index` to a slightly larger integer value.


## 📌 Citation

Please consider citing our work if you find it useful:
```
@article{han2025chords,
  title={CHORDS: Diffusion Sampling Accelerator with Multi-core Hierarchical ODE Solvers},
  author={Han, Jiaqi and Ye, Haotian and Li, Puheng and Xu, Minkai and Zou, James and Ermon, Stefano},
  journal={arXiv preprint arXiv:2507.15260},
  year={2025}
}
```

## Contact

If you have any question, welcome to contact me at:

Jiaqi Han: jiaqihan@stanford.edu

🧩 We warmly welcome **community contributions** for e.g. supporting more models!

