import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any, Callable

from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

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
        
        (latents, t, guidance, 
        pooled_prompt_embeds, prompt_embeds, 
        text_ids, latent_image_ids, 
        joint_attention_kwargs,
        image_embeds, negative_image_embeds, 
        do_true_cfg,
        negative_pooled_prompt_embeds, negative_prompt_embeds,
        guidance_scale, true_cfg_scale, 
        idx) = ret
        
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

        if image_embeds is not None:
            self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents.to(device),
            timestep=timestep.to(device) / 1000,
            guidance=guidance.to(device),
            pooled_projections=pooled_prompt_embeds.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            txt_ids=text_ids.to(device),
            img_ids=latent_image_ids.to(device),
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        if do_true_cfg:
            if negative_image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
            neg_noise_pred = self.transformer(
                hidden_states=latents.to(device),
                timestep=timestep.to(device) / 1000,
                guidance=guidance.to(device),
                pooled_projections=negative_pooled_prompt_embeds.to(device),
                encoder_hidden_states=negative_prompt_embeds.to(device),
                txt_ids=text_ids.to(device),
                img_ids=latent_image_ids.to(device),
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
    
        del ret
        mp_queues[1].put((noise_pred, idx),)


@torch.no_grad()
def forward_chords(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_ip_adapter_image: Optional[PipelineImageInput] = None,
    negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
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
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
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
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
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
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.16),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
        negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
    ):
        negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
    elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
        negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
    ):
        ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

    if self.joint_attention_kwargs is None:
        self._joint_attention_kwargs = {}

    image_embeds = None
    negative_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
        negative_image_embeds = self.prepare_ip_adapter_image_embeds(
            negative_ip_adapter_image,
            negative_ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    
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

    # 6. Denoising loop
    while True:
        allocation = algorithm.get_allocation()
        if allocation == []:
            break
        computed_scores = {}
        for thread_id, (t, k, latents) in enumerate(allocation[:-1]):
            # send to worker
            mp_queues[0].put((latents, timesteps[t], guidance, 
                                pooled_prompt_embeds, prompt_embeds, 
                                text_ids, latent_image_ids, 
                                joint_attention_kwargs,
                                image_embeds, negative_image_embeds, 
                                do_true_cfg,
                                negative_pooled_prompt_embeds, negative_prompt_embeds,
                                guidance_scale, true_cfg_scale, 
                                thread_id))
        # Current thread handles the last one
        t, k, latents = allocation[-1]

        if image_embeds is not None:
            self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timesteps[t].expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents.to(device),
            timestep=timestep.to(device) / 1000,
            guidance=guidance.to(device),
            pooled_projections=pooled_prompt_embeds.to(device),
            encoder_hidden_states=prompt_embeds.to(device),
            txt_ids=text_ids.to(device),
            img_ids=latent_image_ids.to(device),
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        if do_true_cfg:
            if negative_image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
            neg_noise_pred = self.transformer(
                hidden_states=latents.to(device),
                timestep=timestep.to(device) / 1000,
                guidance=guidance.to(device),
                pooled_projections=negative_pooled_prompt_embeds.to(device),
                encoder_hidden_states=negative_prompt_embeds.to(device),
                txt_ids=text_ids.to(device),
                img_ids=latent_image_ids.to(device),
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

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
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        if not full_return:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image_list = []
            latents_list = []
            for latent in hit_x:
                latents_list.append(latent.clone().cpu())
                latent = self._unpack_latents(latent, height, width, self.vae_scale_factor)
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

    return FluxPipelineOutput(images=image), stats
