import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import get_device_states, set_device_states

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


# Source: https://github.com/luyug/GradCache/blob/main/src/grad_cache/context_managers.py
class RandContext:
    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(
            devices=self.fwd_gpu_devices,
            enabled=True,
        )
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None

def is_using_ddp():
    return dist.is_available() and dist.is_initialized()

def ddp_loss_averaging_correction_factor():
    return dist.get_world_size() if is_using_ddp() else 1

def gather_with_local_grad(x):
    if not is_using_ddp():
        return x

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]

    # if from another gpu, detach from graph
    dist.all_gather(gathered, x.detach())
    # use autograd for this gpu
    gathered[rank] = x
    return torch.cat(gathered, dim=0)

def gather_without_local_grad(x):
    if not is_using_ddp():
        return x

    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous())
    return torch.cat(gathered, dim=0)

# Large logical batches are good for metric learning because the objective is to move embeddings towards a high number of distinct points.
# We don't use gradient accumulation because microbatches must be able to look forward to 'future' embeddings to compute loss.
# We don't use the cross batch memory function in Pytorch Metric Learning so that there is no parameter drift with logical-batch embeddings.
# Inspiration: https://github.com/luyug/GradCache/blob/main/src/grad_cache/grad_cache.py
# Inspiration: https://github.com/jahuerta92/star/blob/main/model.py
# First do a no-grad forward pass (saving RNG state for dropout) in order to cache embeddings=Z for the whole logical batch.
# Compute the complete full-batch loss on those cached embeddings and save G = dL/dZ to grad_cache where L=loss.
# Then activations are saved to the computational graph for chunk x/gradient g/embedding z with forward pass z = pl_module(**x).
# The dot product l = z * g is the per-chunk-backprop-loss where dl/dz = g, used for backprop through the model parameters of function z=f(x).
# Note: batch size n must be an exact multiple of microbatch_size m and pl_module(**inputs) must return embeddings.
class ChunkedTrainStep(nn.Module):
    def __init__(self, loss_fn, microbatch_size, miner=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.microbatch_size = microbatch_size
        self.miner = miner

    def forward(self, pl_module, inputs, target):
        n = next(iter(inputs.values())).shape[0]
        m = self.microbatch_size

        states, embeddings_list = [], []
        for i in range(0, n, m):
            x = {k: v[i:i + m] for k, v in inputs.items()}
            states.append(RandContext(*x.values()))
            with torch.inference_mode():
                embeddings_list.append(pl_module(**x).detach())

        # Detach, then use for backprop: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html
        embeddings = torch.cat(embeddings_list, dim=0).detach().requires_grad_()
        del embeddings_list
        embeddings_global = gather_with_local_grad(embeddings)
        target_global = gather_without_local_grad(target)
        indices_tuple = self.miner(embeddings_global, target_global) if self.miner is not None else None
        # Because of requires_grad_() above, PyTorch keeps track of operations in loss_fn that use embeddings.
        loss = self.loss_fn(embeddings_global, target_global, indices_tuple)
        # This allows PyTorch to compute the gradient of the loss (L) with respect to the embeddings (Z).
        # Docs: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
        # Intuition: https://medium.com/@rizqinur2010/partial-derivatives-chain-rule-using-torch-autograd-grad-a8b5917373fa
        grad_cache = torch.autograd.grad(loss, embeddings, retain_graph=False, create_graph=False)[0].detach() # grad returns a 1-tuple (dL/dZ,)
        loss_out = loss.detach()
        del loss, embeddings, embeddings_global, target_global, indices_tuple

        opt = pl_module.optimizers()
        scheduler = pl_module.lr_schedulers()
        opt.zero_grad(set_to_none=True)

        chunks = list(zip(range(0, n, m), states, grad_cache.split(m)))
        for chunk_idx, (i, state, g) in enumerate(chunks):
            x = {k: v[i:i + m] for k, v in inputs.items()}
            sync_grad = chunk_idx == len(chunks) - 1 # only sync on the last one
            with opt.toggle_model(sync_grad=sync_grad):
                with state:
                    z = pl_module(**x)
                    per_chunk_backprop_loss = torch.dot(z.flatten(), g.flatten())
                    per_chunk_backprop_loss *= ddp_loss_averaging_correction_factor()
                    pl_module.manual_backward(per_chunk_backprop_loss)

        opt.step()
        scheduler.step()
        return loss_out
