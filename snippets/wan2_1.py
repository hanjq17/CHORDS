import torch
import numpy as np
from typing import Union, Optional, Dict, Any, List, Callable


from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.pipelines.wan.pipeline_wan import WanPipeline

from algorithms.chords import CHORDS


@torch.no_grad()
def forward_chords_worker(
    self, 
    mp_queues: Optional[torch.FloatTensor] = None,
    device: Optional[str] = None,
    **kwargs,
):

    while True:
        ret = mp_queues[0].get()
        if ret is None:
            del ret
            return
        
        (latents, t, prompt_embeds, negative_prompt_embeds, do_classifier_free_guidance,
            guidance_scale, attention_kwargs, idx) = ret

        latent_model_input = latents.to(self.transformer.dtype)
        timestep = t.expand(latents.shape[0])

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(device),
            timestep=timestep.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_uncond = self.transformer(
                hidden_states=latent_model_input.to(device),
                timestep=timestep.to(device),
                encoder_hidden_states=negative_prompt_embeds.to(device),
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
    
        del ret
        mp_queues[1].put((noise_pred, idx),)



@torch.no_grad()
def forward_chords(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    num_cores: int = 1,
    mp_queues: Optional[torch.FloatTensor] = None,
    full_return: bool = False,
    init_t: str = None,
    stopping_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
):

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    if num_frames % self.vae_scale_factor_temporal != 1:
        print(
            f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = self._execution_device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    # Customize the solver for Algorithm
    def solver(x_t, score_t, t_step, s_step):
        target_value = self.scheduler.custom_step_flexible(
            model_output=score_t,
            step_index=t_step,
            step_index_target=s_step,
            sample=x_t,
            return_dict=False,
        )[0]
        return target_value  # This dtype conversion is important
    
    algorithm = CHORDS(
        T=num_inference_steps,
        x0=latents,
        num_cores=num_cores,
        solver=solver,
        init_t=init_t,
        stopping_kwargs=stopping_kwargs,
        verbose=verbose,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    stats = {}

    start.record()

    pass_count = 0

    while True:
        allocation = algorithm.get_allocation()
        if allocation == []:
            break
        computed_scores = {}
        for thread_id, (t, k, latents) in enumerate(allocation[:-1]):
            # send to worker
            mp_queues[0].put(
                (latents, timesteps[t], prompt_embeds,
                 negative_prompt_embeds, self.do_classifier_free_guidance,
                 guidance_scale,
                 attention_kwargs, thread_id)
            )
        # Current thread handles the last one
        t, k, latents = allocation[-1]
        latent_model_input = latents.to(transformer_dtype)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timesteps[t].expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
        )[0]

        if self.do_classifier_free_guidance:
            noise_uncond = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        
        computed_scores[len(allocation)-1] = noise_pred

        # Collect results from workers
        for _ in range(len(allocation)-1):
            ret = mp_queues[1].get()
            noise_pred, thread_id = ret
            computed_scores[thread_id] = noise_pred.to(device)
        
        # Update algorithm using the computed scores
        scores = []
        for thread_id, (t, k, latents) in enumerate(allocation):
            scores.append((t, k, computed_scores[thread_id]))
        algorithm.update_scores(scores)

        algorithm.update_states(len(allocation))
        delete_ids, earlystop = algorithm.schedule_cores()

        if earlystop:
            break

        algorithm.cur_core_to_compute = algorithm.cur_core_to_compute[len(allocation):]

        if len(delete_ids):
            algorithm.cur_core_to_compute = [core_id for core_id in algorithm.cur_core_to_compute if core_id not in delete_ids]

        pass_count += 1
    
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    time_used = start.elapsed_time(end)

    print("done", flush=True)

    hit_iter_idx, hit_x, hit_time = algorithm.get_last_x_and_hittime()

    stats['flops_count'] = algorithm.get_flops_count()
    stats['pass_count'] = pass_count
    stats['total_time'] = time_used

    # Parse some useful information to stats
    for i in range(len(hit_x)):
        if i > 0:
            diff = torch.linalg.norm(hit_x[i] - hit_x[i-1]).double().item() / hit_x[i].numel()
            stats[f'prev_diff_{hit_iter_idx[i]}'] = diff
        stats[f'hit_time_{hit_iter_idx[i]}'] = hit_time[i]

    print(stats)

    latents = hit_x[-1]

    self._current_timestep = None

    if not output_type == "latent":
        if not full_return:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video_list = []
            latents_list = []
            for latent in hit_x:
                latents_list.append(latent.clone().cpu())
                latent = latent.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latent.device, latent.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                    latent.device, latent.dtype
                )
                latent = latent / latents_std + latents_mean
                video = self.vae.decode(latent, return_dict=False)[0]
                video = self.video_processor.postprocess_video(video, output_type=output_type)
                video_list.append(video)
            video = (video_list, latents_list)

    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    del algorithm
    torch.cuda.empty_cache()

    if not return_dict:
        return (video,)

    return WanPipelineOutput(frames=video), stats
