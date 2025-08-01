import torch
import numpy as np
from typing import Union, Optional, List, Dict, Any, Callable

from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

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
        
        (latents, t, 
            prompt_embeds, pooled_prompt_embeds,
            num_inference_steps,
            do_classifier_free_guidance, guidance_scale, skip_layer_guidance_scale,
            joint_attention_kwargs,
            skip_layer_guidance_start, skip_layer_guidance_stop,
            original_prompt_embeds, original_pooled_prompt_embeds,
            skip_guidance_layers, idx) = ret
        
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(device),
            timestep=timestep.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            pooled_projections=pooled_prompt_embeds.to(device),
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            should_skip_layers = (
                True
                if t > num_inference_steps * skip_layer_guidance_start
                and t < num_inference_steps * skip_layer_guidance_stop
                else False
            )
            if skip_guidance_layers is not None and should_skip_layers:
                timestep = t.expand(latents.shape[0])
                latent_model_input = latents
                noise_pred_skip_layers = self.transformer(
                    hidden_states=latent_model_input.to(device),
                    timestep=timestep.to(device),
                    encoder_hidden_states=original_prompt_embeds.to(device),
                    pooled_projections=original_pooled_prompt_embeds.to(device),
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                    skip_layers=skip_guidance_layers,
                )[0]
                noise_pred = (
                    noise_pred + (noise_pred_text - noise_pred_skip_layers) * skip_layer_guidance_scale
                )
    
        del ret
        mp_queues[1].put((noise_pred, idx),)


@torch.no_grad()
def forward_chords(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
    num_cores: int = 1,
    mp_queues: Optional[torch.FloatTensor] = None,
    full_return: bool = False,
    init_t: str = None,
    stopping_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
):

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    if self.do_classifier_free_guidance:
        if skip_guidance_layers is not None:
            original_prompt_embeds = prompt_embeds
            original_pooled_prompt_embeds = pooled_prompt_embeds
        else:
            original_prompt_embeds = None
            original_pooled_prompt_embeds = None
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // self.transformer.config.patch_size) * (
            width // self.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)
    
    # Customize the solver for Algorithm
    def solver(x_t, score_t, t_step, s_step):
        target_value = self.scheduler.custom_step_flexible(
            model_output=score_t,
            step_index=t_step,
            step_index_target=s_step,
            sample=x_t,
            return_dict=False,
        )[0]
        return target_value.to(x_t.dtype)  # This dtype conversion is important
    
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

    # 7. Denoising loop
    while True:
        allocation = algorithm.get_allocation()
        if allocation == []:
            break
        computed_scores = {}
        for thread_id, (t, k, latents) in enumerate(allocation[:-1]):
            # send to worker
            mp_queues[0].put((latents, timesteps[t], 
                                prompt_embeds, pooled_prompt_embeds,
                                num_inference_steps,
                                self.do_classifier_free_guidance, self.guidance_scale, self._skip_layer_guidance_scale,
                                self.joint_attention_kwargs,
                                skip_layer_guidance_start, skip_layer_guidance_stop,
                                original_prompt_embeds, original_pooled_prompt_embeds,
                                skip_guidance_layers, thread_id))
        # Current thread handles the last one
        t, k, latents = allocation[-1]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timesteps[t].expand(latent_model_input.shape[0])

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(device),
            timestep=timestep.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            pooled_projections=pooled_prompt_embeds.to(device),
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            should_skip_layers = (
                True
                if t > num_inference_steps * skip_layer_guidance_start
                and t < num_inference_steps * skip_layer_guidance_stop
                else False
            )
            if skip_guidance_layers is not None and should_skip_layers:
                timestep = t.expand(latents.shape[0])
                latent_model_input = latents
                noise_pred_skip_layers = self.transformer(
                    hidden_states=latent_model_input.to(device),
                    timestep=timestep.to(device),
                    encoder_hidden_states=original_prompt_embeds.to(device),
                    pooled_projections=original_pooled_prompt_embeds.to(device),
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    skip_layers=skip_guidance_layers,
                )[0]
                noise_pred = (
                    noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                )
        
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

    if output_type == "latent":
        image = latents
    else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        if not full_return:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image_list = []
            latents_list = []
            for latent in hit_x:
                latents_list.append(latent.clone().cpu())
                latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latent, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
                image_list.append(image)
            image = (image_list, latents_list)

    # Offload all models
    self.maybe_free_model_hooks()

    del algorithm
    torch.cuda.empty_cache()

    if not return_dict:
        return (image,)

    return StableDiffusion3PipelineOutput(images=image), stats
    