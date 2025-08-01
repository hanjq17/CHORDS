import torch
import numpy as np
from typing import Union, Optional, Dict, Any, List, Callable

from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE, retrieve_timesteps

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
        
        (latents, t, prompt_embeds, prompt_attention_mask, pooled_prompt_embeds,
            guidance, attention_kwargs, idx) = ret
        
        latent_model_input = latents.to(self.transformer.dtype)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(device),
            timestep=timestep.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            encoder_attention_mask=prompt_attention_mask.to(device),
            pooled_projections=pooled_prompt_embeds.to(device),
            guidance=guidance.to(device),
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
    
        del ret
        mp_queues[1].put((noise_pred, idx),)



@torch.no_grad()
def forward_chords(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Union[str, List[str]] = None,
    height: int = 720,
    width: int = 1280,
    num_frames: int = 129,
    num_inference_steps: int = 50,
    sigmas: List[float] = None,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
    max_sequence_length: int = 256,
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
        prompt_2,
        height,
        width,
        prompt_embeds,
        callback_on_step_end_tensor_inputs,
        prompt_template,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
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
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_template=prompt_template,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        device=device,
        max_sequence_length=max_sequence_length,
    )

    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
    )

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    # for older diffusers versions, use num_latent_frames as input to self.prepare_latents
    # num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        # num_latent_frames,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    # 6. Prepare guidance condition
    guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

    # 7. Denoising loop
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
                 prompt_attention_mask, pooled_prompt_embeds,
                 guidance, attention_kwargs, thread_id)
            )
        # Current thread handles the last one
        t, k, latents = allocation[-1]
        latent_model_input = latents.to(transformer_dtype)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timesteps[t].expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(device),
            timestep=timestep.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            encoder_attention_mask=prompt_attention_mask.to(device),
            pooled_projections=pooled_prompt_embeds.to(device),
            guidance=guidance.to(device),
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

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

    if not output_type == "latent":
        if not full_return:
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video_list = []
            latents_list = []
            for latent in hit_x:
                latents_list.append(latent.clone().cpu())
                latent = latent.to(self.vae.dtype) / self.vae.config.scaling_factor
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

    return HunyuanVideoPipelineOutput(frames=video), stats
