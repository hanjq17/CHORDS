import hydra
import types
from omegaconf import OmegaConf

import torch
from diffusers import (
    HunyuanVideoPipeline,
    WanPipeline, AutoencoderKLWan,
    CogVideoXPipeline, CogVideoXDDIMScheduler,
)

from diffusers.utils import export_to_video

import json

import torch.multiprocessing as mp
import time
import os


def generate_video(
    rank,
    total_ranks,
    queues,
    prompt_file: str,
    output_base_path: str,
    model_config: dict = None,
    algo_config: dict = None,
):
    dtype = torch.float16 if model_config.dtype == "float16" else torch.bfloat16

    # Load models
    if model_config.model_name == 'hunyuan':
        pipe = HunyuanVideoPipeline.from_pretrained(model_config.model_path, torch_dtype=dtype)
        from snippets.hunyuan import (
            forward_chords, forward_chords_worker,
        )
        from snippets.scheduler import custom_step_Euler as custom_step_flexible
    elif model_config.model_name == 'wan2-1':
        vae = AutoencoderKLWan.from_pretrained(model_config.model_path, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_config.model_path, vae=vae, torch_dtype=torch.bfloat16)
        from snippets.wan2_1 import (
            forward_chords, forward_chords_worker,
        )
        from snippets.scheduler import custom_step_Euler as custom_step_flexible
    elif model_config.model_name == 'cogvideo':
        pipe = CogVideoXPipeline.from_pretrained(model_config.model_path, torch_dtype=dtype)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        from snippets.cogvideox import (
            forward_chords, forward_chords_worker,
        )
        from snippets.scheduler import custom_step_DDIM as custom_step_flexible
    else:
        raise NotImplementedError(f"Model path {model_config.model_name} not implemented")
    
    pipe.forward_chords = types.MethodType(forward_chords, pipe)
    pipe.forward_chords_worker = types.MethodType(forward_chords_worker, pipe)
    pipe.scheduler.custom_step_flexible = types.MethodType(custom_step_flexible, pipe.scheduler)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Handle algorithms
    if algo_config.algo_name == 'sequential':
        master_func = pipe.__call__
        worker_func = None
    elif algo_config.algo_name == 'chords':
        master_func = pipe.forward_chords
        worker_func = pipe.forward_chords_worker
    else:
        raise NotImplementedError(f"Method {algo_config.algo_name} not implemented")
    
    algo_kwargs = {key: value for key, value in algo_config.items() if key != 'algo_name'}

    # warmup_prompt = ["A happy grey Maine Coon is enjoying the sunlight on the lawn in a city park"]
    warmup_prompt = []

    if rank > 0:
        # Workers
        pipe = pipe.to(f"cuda:{rank}")
        print('Starting worker at', pipe.device, flush=True)
        if worker_func is not None:
            worker_func(
                mp_queues=queues,
                device=f"cuda:{rank}",
                num_inference_steps=model_config.num_inference_steps,
            )
        print('Shutting down worker at', pipe.device, flush=True)
    else:
        # Master process
        pipe = pipe.to(f"cuda:{rank}")
        print('Starting master process at', pipe.device, flush=True)

        # open prompt file and load the prompt in each line
        with open(prompt_file, "r") as f:
            prompts = f.read().splitlines()
            
        print(prompts, flush=True)

        # warmup
        prompts = warmup_prompt + prompts

        for prompt_idx, prompt in enumerate(prompts):
            print(f"Processing prompt {prompt_idx}/{len(prompts)}: {prompt}", flush=True)

            cur_name = prompt.replace(" ", "_").replace(".", "_")
            cur_output_path = f"{output_base_path}/{cur_name}"
            os.makedirs(cur_output_path, exist_ok=True)

            extra_kwargs = dict(
                mp_queues=queues,
            ) if worker_func is not None else {}

            # Special for CogVideoX
            if model_config.model_name == 'cogvideo':
                extra_kwargs['use_dynamic_cfg'] = model_config.use_dynamic_cfg

            output = master_func(
                height=model_config.height,
                width=model_config.width,
                prompt=prompt,
                num_videos_per_prompt=model_config.num_videos_per_prompt,
                num_inference_steps=model_config.num_inference_steps,
                num_frames=model_config.num_frames,
                guidance_scale=model_config.guidance_scale,
                generator=torch.Generator().manual_seed(model_config.seed),
                **extra_kwargs,
                **algo_kwargs,
            )

            if prompt_idx < len(warmup_prompt):
                print(f"Warmup {prompt_idx}, skipping", flush=True)
                continue
            
            if algo_config.algo_name == 'chords':
                output, stats = output
                print(stats)
                stats["prompt"] = prompt
                stats["ngpu"] = total_ranks
                model_config_dict = OmegaConf.to_container(model_config, resolve=True)
                algo_config_dict = OmegaConf.to_container(algo_config, resolve=True)
                stats["model_config"] = model_config_dict
                stats["algo_config"] = algo_config_dict
                stats["cur_output_path"] = cur_output_path

                # save stats
                with open(f'{cur_output_path}/stats.json', 'w') as f:
                    json.dump(stats, f)

            fps = model_config.fps

            if algo_config.algo_name in ['chords'] and algo_config.full_return:
                video_list, latents_list = output.frames
                for index, video in enumerate(video_list):
                    video_generate = video[0]
                    export_to_video(video_generate, f'{cur_output_path}/video_iter{index}.mp4', fps=fps)
                # save latents
                torch.save(latents_list, f'{cur_output_path}/latents.pt')
            else:
                video_generate = output.frames[0]
                export_to_video(video_generate, f'{cur_output_path}/video.mp4', fps=fps)

            print("done")
            time.sleep(1)

        # shutdown workers
        for _ in range(total_ranks):
            queues[0].put(None)


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="base",
)
def main(config):

    print(config)

    model_config = config.model
    print(model_config)

    algo_config = config.algo
    print(algo_config)

    mp.set_start_method('spawn', force=True)
    queues = mp.Queue(), mp.Queue(), mp.Queue()

    processes = []
    num_processes = max(1, config.ngpu)

    for rank in range(0, num_processes):
        print(f"Starting process {rank}", flush=True)
        p = mp.Process(
            target=generate_video,
            args=(rank, num_processes, queues),
            kwargs=dict(
                prompt_file=config.prompt_file,
                output_base_path=config.output_base_path,
                model_config=model_config,
                algo_config=algo_config,
                )
            )
        p.start()
        processes.append(p)

    queues[2].put(None)

    for p in processes:
        p.join()    # wait for all subprocesses to finish


if __name__ == "__main__":
    main()
