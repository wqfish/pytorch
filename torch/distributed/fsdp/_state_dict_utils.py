import functools
import math
import warnings
from typing import Any, Callable, cast, Dict

import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper

# Import the entire FSDP file to avoid circular imports
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
)
from torch.distributed.fsdp._common_utils import (
    clean_tensor_name,
    FSDP_PREFIX,
    TrainingState,
)
from torch.distributed.fsdp._runtime_utils import _clear_grads_if_needed
from torch.distributed.utils import _replace_by_prefix

from ._fsdp_extensions import (
    _ext_chunk_tensor,
    _ext_pre_load_state_dict_transform,
    _extensions as _user_extensions,
)
from .flat_param import FlatParamHandle


def _enter_full_param_ctx(
    module,
    recurse: bool = False,
    writeback: bool = False,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
) -> None:
    """
    state_dict hooks cannot use the pure context call as the checkpoint flow
    requires to enter the context in the pre-hook but leave the context in the
    post-hook. This API enters the context of ``summon_full_params``.
    """
    assert module._full_param_ctx is None, (
        "Entering the ``summon_full_params`` context but module._full_param_ctx "
        "is not None."
    )
    assert module.training_state != TrainingState.SUMMON_FULL_PARAMS, (
        "Entering the summon_full_params context but the state is already "
        "SUMMON_FULL_PARAMS."
    )
    module._full_param_ctx = module._summon_full_params(
        recurse=recurse,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    )
    module._full_param_ctx.__enter__()


def _exit_full_param_ctx(module) -> None:
    """A helper function to exit ``summon_full_params`` context."""
    module._assert_state([TrainingState.SUMMON_FULL_PARAMS])
    assert module._full_param_ctx is not None
    module._full_param_ctx.__exit__(None, None, None)
    module._full_param_ctx = None


def _common_pre_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """Performs the pre-state_dict tasks shared by all state_dict types."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    module._lazy_init()
    # TODO: change to this call after pre_state_dict_hook is in `nn.Module`.
    # if module.is_root:
    #    _clear_grads_if_needed(module._fsdp_handles(module))
    if module._has_params:
        _clear_grads_if_needed([module._handles[0]])


def _common_summon_pre_state_dict_hook(
    module,
    offload_to_cpu: bool,
    rank0_only: bool,
) -> None:
    """
    Performs the pre-state_dict tasks shared by all state_dict types that require
    ``summon_full_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this hook.
    """
    _enter_full_param_ctx(
        module,
        recurse=False,
        writeback=False,
        offload_to_cpu=offload_to_cpu,
        rank0_only=rank0_only,
    )
    # Since buffers are not sharded and stay casted, restore them to their
    # original user module specified types for checkpoint. We take care to
    # recast in post_state_dict_hook for consistency with the fact that
    # buffers stay casted after forward/backward. We must have the
    # call here instead of above because _summon_full_params calls _lazy_init()
    # which would cast the buffers.
    if module._mixed_precision_enabled_for_buffers():
        module._cast_buffers(dtype=module._buffer_name_to_orig_dtype, recurse=False)


# TODO: change to the decorator style. See ``_full_pre_state_dict_hook``.
def _common_summon_post_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
    inner_hook: Callable,
) -> Dict[str, Any]:
    """
    The post-state_dict flow that shared by all state_dict types that require
    ``summon_full_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this
    hook.
    """
    _replace_by_prefix(state_dict, prefix + f"{FSDP_PREFIX}", prefix)
    module._assert_state([TrainingState.SUMMON_FULL_PARAMS])
    # Return early for trivial cases
    if not state_dict or not module._has_params:
        _exit_full_param_ctx(module)
        return state_dict

    # TODO: Once pre_state_dict hook is supported, this pop should be removed.
    # For `use_orig_params=True`, the `FlatParameter` is not registered, so
    # there is no entry in the state dict for it to pop.
    if not module._use_orig_params:
        state_dict.pop(f"{prefix}{fsdp_file.FLAT_PARAM}")

    # If a rank does not have unsharded parameters(when `rank0_only=True`
    # and `rank != 0`), then the rank only needed to participate in the
    # all-gather and does not need to save the # state dict. We simply check
    # rank0_only to ensure this issue.
    rank0_only = (
        module._state_dict_type == fsdp_file.StateDictType.FULL_STATE_DICT
        and cast(fsdp_file.FullStateDictConfig, module._state_dict_config).rank0_only
    )
    # no_fsdp_return means the state_dict returned by this rank should contain
    # only non-FSDP controlled parameters and buffers.
    no_fsdp_return = rank0_only and module.rank != 0
    if no_fsdp_return and not module._use_orig_params:
        for clean_key in module._buffer_names:
            # This is a hack to support activation checkpoint.
            clean_key = clean_key.replace(
                f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
            )
            state_dict.pop(f"{prefix}{clean_key}", None)
        _exit_full_param_ctx(module)
        return state_dict

    # Loop only the parameters saved in this instance's wrapped module to
    # avoid processing buffers.
    for fqn, param_name, module_name in module._param_fqns:
        # TODO: remove the parameter retrieval. See ``_full_pre_state_dict_hook``.
        param = functools.reduce(getattr, fqn.split("."), module.module)
        fqn = f"{prefix}{fqn}"
        if no_fsdp_return:
            state_dict.pop(fqn)
            continue
        state_dict[fqn] = param
        assert fqn in state_dict, (
            f"FSDP assumes {fqn} is in the state_dict but the state_dict only "
            f"has {state_dict.keys()}. "
            f"prefix={prefix}, module_name={module_name}, "
            f"param_name={param_name} rank={module.rank}."
        )

        inner_hook(module, state_dict, prefix, fqn)
    _exit_full_param_ctx(module)

    # Offload the buffer to CPU if needed -- we do not do this in
    # `_summon_full_params()` since without care, that would free
    # the original buffer's GPU memory and require reallocating
    # that memory later; this only affects the state dict's buffer
    # variable and leaves the original buffer's GPU memory intact
    cpu_device = torch.device("cpu")
    for clean_key in module._buffer_names:
        # This is a hack to support activation checkpoint.
        clean_key = clean_key.replace(f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", "")
        fqn = f"{prefix}{clean_key}"
        if fqn not in state_dict:
            # A buffer can be registered as non-persistent.
            continue
        if no_fsdp_return:
            state_dict.pop(fqn)
        else:
            # TODO: remove the buffer retrieval. See ``_full_pre_state_dict_hook``.
            buf = functools.reduce(getattr, clean_key.split("."), module.module)
            if module._state_dict_config.offload_to_cpu and buf != cpu_device:
                buf = buf.to(cpu_device)
            state_dict[fqn] = buf
    assert module.training_state != TrainingState.SUMMON_FULL_PARAMS
    return state_dict


def _full_pre_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    Hook that runs before model.state_dict() is called. pre-state_dict hook is
    not actually supported by ``nn.Module``. As a result, this API is called
    from ``_full_post_state_dict_hook()`` to simulate the case. Once pre-state_dict
    is supported in ``nn.Module``, this hook will be registered as a hook in
    ``nn.Module``.

    TODO: clean the callsites and hacks after ``pre_state_dict_hook` ` is supported
    in ``nn.Module``.
    """
    _common_pre_state_dict_hook(module, state_dict, prefix)
    _common_summon_pre_state_dict_hook(
        module,
        offload_to_cpu=module._state_dict_config.offload_to_cpu,
        rank0_only=cast(
            fsdp_file.FullStateDictConfig, module._state_dict_config
        ).rank0_only,
    )


def _full_post_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    Hook that runs after model.state_dict() is called before returning result to
    user. For FSDP, we may have to clone the tensors in state_dict as params go
    back to sharded version after _summon_full_params ends, and also remove
    the ``FSDP_WRAPPED_MODULE`` prefix.
    """
    # TODO: remove the hack. See ``_full_pre_state_dict_hook``.
    _full_pre_state_dict_hook(module, state_dict, prefix)

    def inner_hook(
        module,
        state_dict: Dict[str, Any],
        prefix: str,
        fqn: str,
    ) -> None:
        clean_key = fqn
        clean_prefix = clean_tensor_name(prefix)
        # Strip prefix out of key if needed as buffer names and param names
        # do not have prefix considered as they are not computed in `state_dict`
        # call.
        if clean_key.startswith(clean_prefix):
            clean_key = clean_key[len(clean_prefix) :]

        # Clone non-ignored parameters before exiting the
        # `_summon_full_params()` context
        if clean_key not in module._ignored_param_names and not getattr(
            state_dict[fqn], "_has_been_cloned", False
        ):
            try:
                state_dict[fqn] = state_dict[fqn].clone().detach()
                state_dict[fqn]._has_been_cloned = True  # type: ignore[attr-defined]
            except BaseException as e:
                warnings.warn(
                    f"Failed to clone() tensor with name {fqn} on rank {module.rank}. "
                    "This may mean that this state_dict entry could point to invalid "
                    "memory regions after returning from state_dict() call if this "
                    "parameter is managed by FSDP. Please check clone "
                    f"implementation of {fqn}. Error: {str(e)}"
                )

    return _common_summon_post_state_dict_hook(module, state_dict, prefix, inner_hook)


def _full_pre_load_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    _enter_full_param_ctx(module, recurse=False, writeback=True)
    _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")


def _full_post_load_state_dict_hook(module, *args, **kwargs) -> None:
    _exit_full_param_ctx(module)


def _local_pre_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    Hook that runs before model.state_dict() is called. Right now, pre-state_dict
    hook is not supported by the PyTorch core. So this API is called from
    `_local_post_state_dict_hook()` to simulate the case.
    """
    if module._has_params and not module._handles[0].uses_sharded_strategy:
        raise RuntimeError(
            "``local_state_dict`` can only be used when parameters are flatten "
            "and sharded."
        )
    _common_pre_state_dict_hook(module, state_dict, prefix)


def _local_post_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    This hook create a ShardedTensor from the local flat_param and replace
    the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
    will happen. The underlying storage is the same.
    """
    # TODO: remove the hack. See ``_full_pre_state_dict_hook``.
    _local_pre_state_dict_hook(module, state_dict, prefix)

    _replace_by_prefix(state_dict, f"{prefix}{FSDP_PREFIX}", prefix)
    if not module._has_params:
        return state_dict

    # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
    # value as the flat_param but it is a pure Tensor because
    # nn.Module.state_dict() will detach the parameter. Therefore, we need
    # to get flat_param to get the metadata.
    assert module._handles, "Should have returned early"
    flat_param = module._handles[0].flat_param
    # Construct a ShardedTensor from the flat_param.
    full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
    shard_offset = flat_param.numel() * module.rank
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    if valid_data_size > 0 and flat_param._shard_numel_padded > 0:
        flat_param = flat_param.narrow(0, 0, valid_data_size)
    local_shards = [
        Shard.from_tensor_and_offsets(flat_param, [shard_offset], module.rank)
    ]
    sharded_tensor = init_from_local_shards(
        local_shards, full_numel, process_group=module.process_group
    )  # type: ignore[assignment]
    if module._state_dict_config.offload_to_cpu:
        sharded_tensor = sharded_tensor.cpu()
    state_dict[f"{prefix}{fsdp_file.FLAT_PARAM}"] = sharded_tensor
    return state_dict


def _local_post_load_state_dict_hook(module, *args, **kwargs) -> None:
    pass


def _local_pre_load_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
    _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")
    fqn = f"{prefix}{FSDP_PREFIX}{fsdp_file.FLAT_PARAM}"
    if fqn not in state_dict:
        assert not module._has_params, (
            "No `FlatParameter` in `state_dict` for this FSDP instance "
            "but it has parameters"
        )
        return
    load_tensor = state_dict[fqn]
    assert isinstance(
        load_tensor, ShardedTensor
    ), "Tensors in local_state_dict should be ShardedTensor."

    # Convert the ShardedTensor to a Tensor.
    shards = load_tensor.local_shards()
    assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
    load_tensor = shards[0].tensor

    # Get the metadata of the flat_param to decide whether to pad the loaded
    # tensor.
    flat_param = module._handles[0].flat_param
    assert flat_param is not None
    if flat_param._shard_numel_padded not in (0, flat_param.numel()):
        assert load_tensor.numel() < flat_param.numel(), (
            f"Local shard size = {flat_param.numel()} and the tensor in "
            f"the state_dict is {load_tensor.numel()}."
        )
        load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
    state_dict[fqn] = load_tensor


def _sharded_pre_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    Hook that runs before model.state_dict() is called. Check
    ``_full_pre_load_state_dict_hook`` for the detail.
    """
    if module._has_params and not module._handles[0].uses_sharded_strategy:
        raise RuntimeError(
            "``sharded_state_dict`` can only be used when parameters are flatten "
            "and sharded."
        )
    _common_pre_state_dict_hook(module, state_dict, prefix)
    # Setting offload_to_cpu here does not work even if offload_to_cpu is True.
    # We have to create ShardedTensor first then move it to CPU.
    _common_summon_pre_state_dict_hook(
        module,
        offload_to_cpu=False,
        rank0_only=False,
    )


def _sharded_post_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    The hook replaces the unflattened, unsharded parameter in the state_dict
    with a unflattened, sharded parameter (a ShardedTensor).
    """

    # TODO: remove the hack. See ``_full_pre_state_dict_hook``.
    _sharded_pre_state_dict_hook(module, state_dict, prefix)

    def inner_hook(module, state_dict: Dict[str, Any], prefix: str, fqn: str):
        param = state_dict[fqn]
        sharded_tensor = _ext_chunk_tensor(
            tensor=param,
            rank=module.rank,
            world_size=module.world_size,
            num_devices_per_node=torch.cuda.device_count(),
            pg=module.process_group,
        )
        if module._state_dict_config.offload_to_cpu:
            sharded_tensor = sharded_tensor.cpu()
        state_dict[fqn] = sharded_tensor

    return _common_summon_post_state_dict_hook(module, state_dict, prefix, inner_hook)


def _sharded_post_load_state_dict_hook(module, *args, **kwargs) -> None:
    if module._use_orig_params:
        module._register_orig_params()


def _sharded_pre_load_state_dict_hook(
    module,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
    _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")
    if not module._has_params:
        return

    if not module._handles[0].uses_sharded_strategy:
        raise RuntimeError(
            "load_sharded_state_dict can only be called when parameters "
            "are flatten and sharded."
        )

    nonsharded_tensors = []
    shared_fqns = [fqn for fqn, _, _ in module._shared_param_fqns]
    loaded_shapes = []
    for fqn, _, _ in module._param_fqns:
        full_fqn = f"{prefix}{FSDP_PREFIX}{fqn}"
        param = state_dict.pop(full_fqn)
        if fqn in shared_fqns:
            continue
        # All-gather the param (ShardedTensor)
        param, shards = _ext_pre_load_state_dict_transform(param)
        loaded_shapes.append(param.size())
        assert len(shards) < 2, (
            "Expects 0 or 1 shard per rank "
            f"but got {len(shards)} shards on rank {module.rank}."
        )
        param_numel = param.size().numel()
        dim_0_size = param.size()[0]
        chunk_size = (
            math.ceil(dim_0_size / module.world_size) * param_numel // dim_0_size
        )
        if len(shards) == 1:
            local_tensor = shards[0].tensor.flatten()
            if not local_tensor.is_cuda:
                local_tensor = local_tensor.cuda()
            num_padding = chunk_size - local_tensor.numel()
            if num_padding > 0:
                local_tensor = F.pad(local_tensor, [0, num_padding])
        else:
            local_tensor = torch.zeros(chunk_size, dtype=param.dtype).cuda()
        tensor = torch.empty(
            chunk_size * module.world_size, dtype=local_tensor.dtype
        ).cuda()
        dist.all_gather_into_tensor(tensor, local_tensor, group=module.process_group)
        tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
        nonsharded_tensors.append(tensor)

    # Create a new flat_param from the loaded, non-sharded tensors.
    flat_param = module._handles[0].flat_param
    loaded_flat_param = FlatParamHandle.flatten_params(
        nonsharded_tensors, requires_grad=False
    )

    # Get the chunk from the loaded flat_param for the local rank.
    loaded_flat_tensor, num_to_pad = FlatParamHandle._get_shard(
        loaded_flat_param,
        module.rank,
        module.world_size,
    )
    loaded_flat_tensor.to(flat_param.device)
    assert all(s1 == s2 for s1, s2 in zip(loaded_shapes, flat_param._shapes)), (
        f"The original shapes in FSDP are {flat_param._shapes}. "
        f"The loaded shapes are {loaded_shapes}. "
        f"FSDP extension is {'NOT' if _user_extensions is None else ''} None."
    )
    assert flat_param.numel() == loaded_flat_tensor.numel(), (
        f"The loaded local chunk has different numel({loaded_flat_tensor.numel()}) "
        f"from the local chunk {flat_param.numel()}."
    )
    assert flat_param._shard_numel_padded == num_to_pad, (
        f"The loaded local chunk has different padding({num_to_pad}) "
        f"from the local chunk {flat_param._shard_numel_padded}."
    )
    state_dict[f"{prefix}{FSDP_PREFIX}{fsdp_file.FLAT_PARAM}"] = loaded_flat_tensor
    if module._use_orig_params:
        module._deregister_orig_params()


@torch.no_grad()
def _post_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> Dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() of this
    FSDP module is executed. ``module._state_dict_type`` is used to decide
    what postprocessing will be done.
    """
    _post_state_dict_hook_fn = {
        fsdp_file.StateDictType.FULL_STATE_DICT: _full_post_state_dict_hook,
        fsdp_file.StateDictType.LOCAL_STATE_DICT: _local_post_state_dict_hook,
        fsdp_file.StateDictType.SHARDED_STATE_DICT: _sharded_post_state_dict_hook,
    }
    fsdp_module = cast(fsdp_file.FullyShardedDataParallel, module)
    processed_state_dict = _post_state_dict_hook_fn[fsdp_module._state_dict_type](
        fsdp_module, state_dict, prefix
    )
    # Restore buffers, which currently are in their full precision type,
    # back to their mixed precision type. This is because buffers are cast
    # during lazy_init() and stay at their mixed precision type before/after
    # forward/backward. As a result state_dict() should maintain this.
    if fsdp_module._is_root and fsdp_module._mixed_precision_enabled_for_buffers():
        fsdp_module._cast_buffers(recurse=True)
    return processed_state_dict


@torch.no_grad()
def _pre_load_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    """
    ``_pre_state_dict_hook` is called before ``module._load_from_state_dict()``
    is called. ``module._state_dict_type`` is used to decide what preprocessing
    will be done.
    """
    _pre_load_state_dict_hook_fn = {
        fsdp_file.StateDictType.FULL_STATE_DICT: _full_pre_load_state_dict_hook,
        fsdp_file.StateDictType.LOCAL_STATE_DICT: _local_pre_load_state_dict_hook,
        fsdp_file.StateDictType.SHARDED_STATE_DICT: _sharded_pre_load_state_dict_hook,
    }
    # Code that is common for all state_dict impls
    fsdp_module = cast(fsdp_file.FullyShardedDataParallel, module)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Dispatch into state_dict specific implementation of pre-hook.
    _pre_load_state_dict_hook_fn[fsdp_module._state_dict_type](
        fsdp_module, state_dict, prefix
    )


@torch.no_grad()
def _post_load_state_dict_hook(module: nn.Module, *args: Any) -> None:
    _post_load_state_dict_hook_fn = {
        fsdp_file.StateDictType.FULL_STATE_DICT: _full_post_load_state_dict_hook,
        fsdp_file.StateDictType.LOCAL_STATE_DICT: _local_post_load_state_dict_hook,
        fsdp_file.StateDictType.SHARDED_STATE_DICT: _sharded_post_load_state_dict_hook,
    }
    # Code that is common for all state_dict impls
    fsdp_module = cast(fsdp_file.FullyShardedDataParallel, module)
    # Dispatch into state_dict type specific implementation of post-hook for
    # loading state_dict.
    _post_load_state_dict_hook_fn[fsdp_module._state_dict_type](fsdp_module)
