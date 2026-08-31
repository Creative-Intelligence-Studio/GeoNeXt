import hashlib
import json
import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

RESUME_META_FILE = "resume_meta.json"


def _checkpoint_dir(output_path, step):
    return os.path.join(output_path, f"training_state-step-{step}")


def _find_latest_checkpoint_dir(output_path):
    if not os.path.isdir(output_path):
        return None
    candidates = []
    for name in os.listdir(output_path):
        if not name.startswith("training_state-step-"):
            continue
        try:
            step = int(name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        path = os.path.join(output_path, name)
        if os.path.isdir(path):
            candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _save_resume_state(accelerator, output_path, global_step, epoch_id, step_in_epoch):
    accelerator.wait_for_everyone()
    checkpoint_path = _checkpoint_dir(output_path, global_step)
    accelerator.save_state(checkpoint_path)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        meta = {
            "global_step": int(global_step),
            "epoch_id": int(epoch_id),
            "step_in_epoch": int(step_in_epoch),
        }
        with open(os.path.join(checkpoint_path, RESUME_META_FILE), "w", encoding="utf-8") as handle:
            json.dump(meta, handle)


def _load_resume_state(accelerator, resume_from, output_path):
    if resume_from is None:
        return None
    if resume_from == "latest":
        checkpoint_path = _find_latest_checkpoint_dir(output_path)
        if checkpoint_path is None:
            return None
    else:
        checkpoint_path = os.path.abspath(resume_from)
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint folder does not exist: {checkpoint_path}")

    accelerator.load_state(checkpoint_path)
    meta_path = os.path.join(checkpoint_path, RESUME_META_FILE)
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    else:
        meta = {"global_step": 0, "epoch_id": 0, "step_in_epoch": 0}
    meta["checkpoint_path"] = checkpoint_path
    return meta


def _module_resume_fingerprint(unwrapped_model, module_attr="dit", max_tensors=8, sample_size=2048):
    pipe = getattr(unwrapped_model, "pipe", None)
    module = getattr(pipe, module_attr, None) if pipe is not None else None
    if module is None:
        return None

    entries = []
    for name, tensor in sorted(module.state_dict().items()):
        if not torch.is_tensor(tensor):
            continue
        if not tensor.is_floating_point():
            continue
        if tensor.numel() == 0:
            continue
        flat = tensor.detach().float().view(-1)
        n = min(sample_size, flat.numel())
        sample = flat[:n].cpu()
        sample_bytes = sample.numpy().tobytes()
        entries.append({
            "name": name,
            "shape": tuple(tensor.shape),
            "hash": hashlib.sha1(sample_bytes).hexdigest()[:12],
            "mean": float(sample.mean().item()),
            "std": float(sample.std(unbiased=False).item()) if n > 1 else 0.0,
            "norm": float(sample.norm().item()),
        })
        if len(entries) >= max_tensors:
            break
    return {"module": f"pipe.{module_attr}", "entries": entries}


def _compare_fingerprints(before_fp, after_fp):
    if before_fp is None or after_fp is None:
        return None
    before = {entry["name"]: entry["hash"] for entry in before_fp.get("entries", [])}
    after = {entry["name"]: entry["hash"] for entry in after_fp.get("entries", [])}
    common = sorted(set(before.keys()) & set(after.keys()))
    changed = [name for name in common if before[name] != after[name]]
    return {
        "total_common": len(common),
        "changed": changed,
    }


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    resume_from = getattr(args, "resume_from", None) if args is not None else None
    pre_resume_fp = None
    if resume_from is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        pre_resume_fp = _module_resume_fingerprint(unwrapped_model)
        if accelerator.is_main_process and pre_resume_fp is not None:
            print(
                f"[ResumeCheck] Pre-resume fingerprint on {pre_resume_fp['module']} "
                f"(sample_tensors={len(pre_resume_fp['entries'])})"
            )

    resume_meta = _load_resume_state(
        accelerator=accelerator,
        resume_from=resume_from,
        output_path=getattr(args, "output_path", "./models") if args is not None else "./models",
    )
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    if resume_meta is not None:
        global_step = int(resume_meta.get("global_step", 0))
        start_epoch = int(resume_meta.get("epoch_id", 0))
        start_step_in_epoch = int(resume_meta.get("step_in_epoch", 0))
        model_logger.num_steps = global_step
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, "_debug_forward_step"):
            unwrapped_model._debug_forward_step = global_step
        if hasattr(unwrapped_model, "pipe"):
            unwrapped_model.pipe._debug_forward_step = global_step
        if accelerator.is_main_process:
            print(f"[Resume] Loaded state from {resume_meta.get('checkpoint_path')}")
            print(f"[Resume] global_step={global_step}, epoch={start_epoch}, step_in_epoch={start_step_in_epoch}")
            post_resume_fp = _module_resume_fingerprint(unwrapped_model)
            fp_cmp = _compare_fingerprints(pre_resume_fp, post_resume_fp)
            if fp_cmp is not None:
                print(
                    f"[ResumeCheck] Fingerprint changed tensors: "
                    f"{len(fp_cmp['changed'])}/{fp_cmp['total_common']}"
                )
                if len(fp_cmp["changed"]) == 0:
                    print("[ResumeCheck][WARN] No sampled tensor changed after resume load.")
                else:
                    print(f"[ResumeCheck] Changed (first 5): {fp_cmp['changed'][:5]}")
    elif resume_from is not None and accelerator.is_main_process:
        print(f"[Resume] No checkpoint found for resume_from={resume_from}. Starting from scratch.")

    for epoch_id in range(start_epoch, num_epochs):
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch_id + 1}/{num_epochs}", 
            disable=not accelerator.is_main_process,
            leave=True
        )
        for step_in_epoch, data in enumerate(progress_bar):
            if epoch_id == start_epoch and step_in_epoch < start_step_in_epoch:
                continue
            # Skip empty or invalid samples.
            if data is None:
                continue
            
            # Extra guard: skip if the parsed sample has no video payload.
            if data.get("video") is None:
                continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                global_step += 1
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
                if save_steps is not None and global_step % save_steps == 0:
                    _save_resume_state(
                        accelerator=accelerator,
                        output_path=model_logger.output_path,
                        global_step=global_step,
                        epoch_id=epoch_id,
                        step_in_epoch=step_in_epoch + 1,
                    )
                if accelerator.is_main_process:
                    # Read current learning rate.
                    current_lr = 0.0
                    if hasattr(optimizer, "param_groups"):
                        current_lr = optimizer.param_groups[0]['lr']
                    
                    # Refresh progress bar display.
                    progress_bar.set_postfix(
                        loss=f"{loss.item():.4f}", 
                        lr=f"{current_lr:.1e}"
                    )
        start_step_in_epoch = 0
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            _save_resume_state(
                accelerator=accelerator,
                output_path=model_logger.output_path,
                global_step=global_step,
                epoch_id=epoch_id + 1,
                step_in_epoch=0,
            )
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
