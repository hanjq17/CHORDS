import torch
from typing import Union, Tuple, Optional, List, Dict, Any, Callable
import math

from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipelineOutput, retrieve_timesteps
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks

from algorithms.chords import CHORDS


def custom_step_flexible(
    self,
    model_output: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    sample: torch.Tensor,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    variance_noise: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        eta (`float`):
            The weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`, defaults to `False`):
            If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
            because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
            clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
            `use_clipped_model_output` has no effect.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        variance_noise (`torch.Tensor`):
            Alternative to generating noise with `generator` by directly providing the noise for the variance
            itself. Useful for methods such as [`CycleDiffusion`].
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    # prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # To make style tests pass, commented out `pred_epsilon` as it is an unused variable
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        # pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        # pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    a_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5
    b_t = alpha_prod_t_prev**0.5 - alpha_prod_t**0.5 * a_t

    prev_sample = a_t * sample + b_t * pred_original_sample

    if not return_dict:
        return (
            prev_sample,
            pred_original_sample,
        )

    return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)



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
        
        (latents, t, prompt_embeds, image_rotary_emb,
            do_classifier_free_guidance, use_dynamic_cfg, num_inference_steps, guidance_scale,
            attention_kwargs, idx) = ret
        
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0])

        # predict noise model_output
        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            timestep=timestep.to(device),
            image_rotary_emb=[emb.to(device) for emb in image_rotary_emb],
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()

        # perform guidance
        if use_dynamic_cfg:
            self._guidance_scale = 1 + guidance_scale * (
                (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
            )
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
    
        del ret
        mp_queues[1].put((noise_pred, idx),)


@torch.no_grad()
def forward_chords(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 226,
    num_cores: int = 1,
    mp_queues: Optional[torch.FloatTensor] = None,
    full_return: bool = False,
    init_t: str = None,
    stopping_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Union[CogVideoXPipelineOutput, Tuple]:

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
    width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
    num_frames = num_frames or self.transformer.config.sample_frames

    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds,
        negative_prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._interrupt = False

    # 2. Default call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    self._num_timesteps = len(timesteps)

    # 5. Prepare latents
    latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

    # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
    patch_size_t = self.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
        num_frames += additional_frames * self.vae_scale_factor_temporal

    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        if self.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    # Customize the solver for Algorithm
    def solver(x_t, score_t, t_step, s_step):
        target_value = self.scheduler.custom_step_flexible(
            model_output=score_t,
            timestep=timesteps[t_step],
            prev_timestep=timesteps[t_step] - self.scheduler.num_train_timesteps // self.scheduler.num_inference_steps * (s_step - t_step),
            sample=x_t,
            return_dict=False,
        )[0]
        return target_value.to(prompt_embeds.dtype)  # This dtype conversion is important
    
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
                 image_rotary_emb, do_classifier_free_guidance,
                 use_dynamic_cfg, num_inference_steps,
                 guidance_scale, attention_kwargs, thread_id)
            )
        # Current thread handles the last one
        t, k, latents = allocation[-1]

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timesteps[t].expand(latent_model_input.shape[0])

        # predict noise model_output
        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(device),
            encoder_hidden_states=prompt_embeds,
            timestep=timestep.to(device),
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()

        # perform guidance
        if use_dynamic_cfg:
            self._guidance_scale = 1 + guidance_scale * (
                (1 - math.cos(math.pi * ((num_inference_steps - timesteps[t].item()) / num_inference_steps) ** 5.0)) / 2
            )
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

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
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video_list = []
            latents_list = []
            for latent in hit_x:
                latents_list.append(latent.clone().cpu())
                latent = latent[:, additional_frames:]
                video = self.decode_latents(latent)
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)
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

    return CogVideoXPipelineOutput(frames=video), stats