import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler
from accelerate.data_loader import BatchSamplerShard
from pytorch_metric_learning.samplers import MPerClassSampler


def distributed_mperclass_sampler(labels, m, batch_size):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    global_batch_size = batch_size * world
    num_samples = (len(labels) // global_batch_size) * global_batch_size

    sampler = MPerClassSampler(labels, m=m, batch_size=global_batch_size, length_before_new_iter=num_samples)
    # Docs: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler
    # This makes sure the data is divisible by the global batch size
    batches_global = BatchSampler(sampler, batch_size=global_batch_size, drop_last=True)
    # Docs: https://huggingface.co/docs/accelerate/package_reference/torch_wrappers#accelerate.data_loader.BatchSamplerShard
    # This makes sure each gpu sees a contiguous slice of the global batch
    return BatchSamplerShard(batches_global, num_processes=world, process_index=rank, split_batches=True, even_batches=False)

def distributed_eval_sampler(dataset):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    return range(rank, len(dataset), world)
