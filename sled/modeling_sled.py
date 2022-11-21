import abc
import inspect
import os
import warnings
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from makefun import create_function
from requests.exceptions import HTTPError
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoModelForSeq2SeqLM, WEIGHTS_NAME
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging


from .configuration_sled import SledConfig

logger = logging.get_logger(__name__)


PREFIX_KEY = 'prefix_length'

def _legacy_download_weights(kwargs, pretrained_model_name_or_path):
    # noinspection PyUnresolvedReferences
    from transformers.utils import hf_bucket_url, cached_path, EntryNotFoundError
    archive_file = hf_bucket_url(
        pretrained_model_name_or_path,
        filename=WEIGHTS_NAME,
        revision=kwargs.pop("revision", None),
        mirror=kwargs.pop("mirror", None),
        subfolder=kwargs.pop("subfolder", None),
    )
    logger.info(f'Looking for pretrained weights on {archive_file}')

    try:
        # Load from URL or cache if already cached
        user_agent = {"file_type": "model", "framework": "pytorch",
                      "from_auto_class": kwargs.pop("_from_auto", False)}
        from_pipeline = kwargs.pop("_from_pipeline", None)
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline
        resolved_archive_file = cached_path(
            archive_file,
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", False),
            local_files_only=kwargs.pop("local_files_only", False),
            use_auth_token=kwargs.pop("use_auth_token", None),
            user_agent=user_agent,
        )

        logger.info(f'Successfully downloaded the weights to {resolved_archive_file}')
        return torch.load(resolved_archive_file, map_location="cpu")
    except EntryNotFoundError:
        logger.info('Did not find a SLED weights file to be loaded.'
                    ' If this is not unexpected, please reach out. '
                    'Note - loading sharded weight file is not currently supported')
    return None

def _download_weights(kwargs, pretrained_model_name_or_path):
    try:
        from transformers.utils import cached_file, EntryNotFoundError, HUGGINGFACE_CO_RESOLVE_ENDPOINT
        # Load from URL
        user_agent = {"file_type": "model", "framework": "pytorch",
                      "from_auto_class": kwargs.pop("_from_auto", False)}
        cached_filename = cached_file(
            pretrained_model_name_or_path,
            WEIGHTS_NAME,  # shard_filename,
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", False),
            local_files_only=kwargs.pop("local_files_only", False),
            use_auth_token=kwargs.pop("use_auth_token", None),
            user_agent=user_agent,
            revision=kwargs.pop("revision", None),
            subfolder=kwargs.pop("subfolder", None),
            _commit_hash=kwargs.pop("_commit_hash", None),
        )
        logger.info(f'Successfully downloaded the weights to {cached_filename}')
        return torch.load(cached_filename, map_location="cpu")
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
    except ImportError:
        # probably an older version of transformers. try to fallback on legacy load
        logger.info('Could not find a weights file with the new huggingface api hub. Attempting fallback to the old api')
    except (OSError, EntryNotFoundError) as e:
        logger.info('Did not find a SLED weights file to be loaded.'
                    ' If this is not unexpected, please reach out. '
                    f'Note - loading sharded weight file is not currently supported ({e})')
        return None
    except HTTPError:
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {WEIGHTS_NAME}. You should try"
            " again after checking your internet connection."
        )

    try:
        return _legacy_download_weights(kwargs, pretrained_model_name_or_path)  # attempt fallback
    except ImportError:
        logger.error('Could not find a SLED weights file to be loaded due to unmatching transformers version. '
                     'Please open as issue on https://github.com/Mivg/SLED/issues')
        raise


def _find_tensor_inds_and_size(*args, **kwargs):
    args_tensor_inds = [i for i, v in enumerate(args) if isinstance(v, torch.Tensor)]
    kwargs_tensor_keys = [k for k, v in kwargs.items() if isinstance(v, torch.Tensor)]

    assert len(args_tensor_inds) + len(kwargs_tensor_keys) > 0, "no tensors were found"
    # tensor are 2d at this point with [N, s], N is the number of inputs, s is the (padded) sequence length
    size = args[args_tensor_inds[0]].size() if len(args_tensor_inds) > 0 else kwargs[kwargs_tensor_keys[0]].size()
    assert len(size) == 2, f"expected 2-d tensors but got tensors of shape: {size}"
    _, s = size
    # can also assert that all tensors here share the same size, but we will skip this for efficiency’s sake and make
    # this assumption

    return args_tensor_inds, kwargs_tensor_keys, s


def _pad_if_needed(v, pad=None):
    if pad is None:
        return v
    return torch.pad


def _concat_input_prefix_if_needed(v, start_ind, end_ind, prefix_length=None, pad=None):
    if prefix_length is None:
        v = v[:, start_ind:end_ind]
        if pad is not None and pad != 0:
            # assert v.dim() == 2
            assert pad >= 0, f'padding should be non negative but it is negative ({pad})'
            v = F.pad(v, (0, pad), "constant", 0)
            # padding only from the right so (0,pad), and only in the second axis so not length 4
        return v
    return torch.cat((v[:, :prefix_length], v[:, prefix_length + start_ind:prefix_length + end_ind]), axis=1)


def _fix_args(args, args_tensor_inds, start_ind, end_ind, prefix_length=None, pad=None):
    return tuple(v if i not in args_tensor_inds else
                 _concat_input_prefix_if_needed(v, start_ind, end_ind, prefix_length, pad)
                 for i, v in enumerate(args))


def _fix_kwargs(kwargs, kwargs_tensor_keys, start_ind, end_ind, prefix_length=None, pad=None):
    return {
        k: v if (k not in kwargs_tensor_keys or k.startswith("decoder")) else
        _concat_input_prefix_if_needed(v, start_ind, end_ind, prefix_length, pad)
        for k, v in kwargs.items()
    }


def _stack_args(stack_args, args_tensor_inds):
    return tuple(v if i not in args_tensor_inds else torch.cat(tuple(si[i] for si in stack_args))
                 for i, v in enumerate(stack_args[0]))


def _stack_kwargs(stack_kwargs, kwargs_tensor_keys):
    try:
        return {k: v if k not in kwargs_tensor_keys else torch.cat(tuple(si[k] for si in stack_kwargs))
                for k, v in stack_kwargs[0].items()}
    except RuntimeError as e:
        for k in kwargs_tensor_keys:
            logger.warning(f'problematic key={k}. size={tuple(si[k].size() for si in stack_kwargs)}')
        if str(e).startswith('Sizes of tensors must match except in dimension 0'):
            logger.warning('Most likely you passed in non-padded batch. make sure all examples in the batch have the same length')
        raise


def _unstack_encoder_outputs(stacked_output, n, bs):
    if isinstance(stacked_output, tuple):
        return [tuple(v if not isinstance(v, torch.Tensor) else v[i * bs:(i + 1) * bs] for v in stacked_output)
                for i in range(n)]
    # works for dict as well as structured outputs
    return [type(stacked_output)(**{k: v if not isinstance(v, torch.Tensor) else v[i*bs:(i+1)*bs]
                                    for k, v in stacked_output.items()})
            for i in range(n)]


def _extract_keyword_args(kwargs, arg_names, prefix=None):
    new_kwargs = {arg_name: kwargs.get(arg_name, None) for arg_name in arg_names}
    if prefix is not None:
        for arg_name in arg_names:
            new_arg_name = prefix + arg_name
            if new_arg_name in kwargs and new_arg_name not in new_kwargs:
                new_kwargs[arg_name] = kwargs[new_arg_name]
    return new_kwargs


def _slice_tensor(v, start, end, prefix_length=None):
    prefix_length = prefix_length or 0
    return v[:, start+prefix_length:end+prefix_length]


def _merge_encoder_outputs(encoder_outputs_list):
    # a list of 4-tuples, first value is the returned value from the encoder, then the start and end indices inside the
    # tensors that we should take, and finally prefix_length (None if was not used)

    # presumed order of returned tuple from encoders: last_hidden_state, hidden_states, attentions

    # the first output, as returned by the underlying model on the first window
    resulting_output = encoder_outputs_list[0][0]
    if isinstance(resulting_output, tuple):  # not in return dict mode:
        resulting_list = []
        for i in range(len(resulting_output)):
            if resulting_output[i] is None:
                resulting_list.append(None)
            elif (
                    isinstance(resulting_output[i], (int, float, tuple, MockTensorForConditionalGeneration))
                    or resulting_output[i].dim() != 3
            ):
                continue
            else:
                assert isinstance(resulting_output[i], torch.Tensor)
                # tensors are of of size (N, w, d), N the batch size, w the current window size and d the hidden
                # state size/logits dimension these are the only parts in the encoder output that we need to merge
                # between windows
                resulting_list.append(
                    torch.cat(tuple(_slice_tensor(out[i], start, end, prefix_length)
                                    for out, start, end, prefix_length in encoder_outputs_list), dim=1)
                )  # requires extra GPU memory because it doesn't dump the old copy of the tensors yet
        resulting_output = tuple(resulting_list)
    else:
        for key in resulting_output.keys():
            if resulting_output[key] is None:
                continue
            if isinstance(resulting_output[key], tuple):
                resulting_output[key] = None  # encoder outputs are not tuples, only the decoders
            else:
                assert isinstance(resulting_output[key], torch.Tensor)
                if resulting_output[key].dim() != 3:
                    continue  # decoder outputs may be 4d tensors
                # tensors are of of size (N, w, d), N the batch size, w the current window size and d the hidden
                # state size/logits dimension
                resulting_output[key] = torch.cat(
                    tuple(_slice_tensor(out[key], start, end, prefix_length)
                          for out, start, end, prefix_length in encoder_outputs_list), dim=1
                )

    return resulting_output


class MockDecoder(nn.Module):
    def forward(self, *_, **__):
        return tuple()

    def to(self, *_, **__):
        return self


class SledPretrainedModel(PreTrainedModel, metaclass=abc.ABCMeta):
    config_class = SledConfig
    auto_model_loader = AutoModel
    IGNORE_CONFIG_KEYS = {'model_type', '_name_or_path'}  # config keys we allow to be mismatched between the
    # SledConfig and the underlying model's config

    def __init__(self, underlying_model: PreTrainedModel, config: SledConfig):
        """

        :param underlying_model: The backbone model to use.
                                Warning - once given, it should not be used directly, as it may cause unexpected behaviours
        :param config:
        """
        super(SledPretrainedModel, self).__init__(config)
        self._underlying_model = (
            underlying_model  # crucial this will be before any calls to members that is in the base model
        )

        self._window_fraction = config.window_fraction
        self._context_size = config.context_size
        self._window_margin = int(config.context_size * (1 - config.window_fraction) / 2)
        self._sliding_method = config.sliding_method or 'dynamic'
        assert self._sliding_method in {'loop', 'stacked', 'dynamic', 'decoderonly'}

        for override_member in ['is_parallelizable', 'supports_gradient_checkpointing']:
            setattr(self, override_member, getattr(underlying_model, override_member))

        # setting the base_model_prefix to return the correct underlying model and link to some methods
        # implemented in the base
        self.base_model_prefix = "sled_base_model_prefix"
        self.sled_base_model_prefix = self._underlying_model.base_model

        # override generation preparation functions that may be overridden by underlying models but will be
        # found in our wrapper. We wished we could do it a follows:
        # for method_name, _ in inspect.getmembers(PreTrainedModel, predicate=inspect.isfunction):
        #     if method_name not in {"_replicate_for_data_parallel", 'modules'}:
        #         setattr(self, method_name, getattr(underlying_model, method_name))
        # However, the above is too broad and dangerous, so we will do it directly
        for method_name in {"_init_weights", "prepare_inputs_for_generation"}:
            if hasattr(underlying_model, method_name):
                setattr(self, method_name, getattr(underlying_model, method_name))

        # set the resize_token_embeddings
        vocab_size = underlying_model.get_input_embeddings().weight.size(0)
        assert hasattr(self.config, 'vocab_size'), 'Underlying models must have a vocab_size config'
        assert underlying_model.config.vocab_size == vocab_size
        self.resize_token_embeddings(vocab_size)  # the underlying model may have a different vocab size compared to its base config

        self._verified_config_consistency = False
        self._verify_config_consistency()
        self._verified_config_consistency = False  # We would like to do it later again (before the first forward)

        self._prepend_prefix = config.prepend_prefix
        self._encode_prefix = config.encode_prefix

        # now, let's create the forward function
        self._create_forward_function()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        assert isinstance(pretrained_model_name_or_path, str), 'pretrained_model_name_or_path must be a path to a ' \
                                                               'checkpoint or a local config file (json)'
        config = kwargs.pop('config', None) or \
                 cls.config_class.from_pretrained(pretrained_model_name_or_path,
                                                  use_auth_token=kwargs.get('use_auth_token', False))
        state_dict = kwargs.pop('state_dict', None)
        if os.path.exists(pretrained_model_name_or_path):
            # if pretrained_model_name_or_path is a saved checkpoint
            if pretrained_model_name_or_path.endswith('json'):
                underlying_model = cls.auto_model_loader.from_pretrained(config.underlying_config, *model_args, **kwargs)
            else:
                # otherwise, it is a config json path + weights. Note LSED doesn't have any weights of its own,
                # so the state dict is only for the underlying model
                backbone_state_dict = torch.load(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME), map_location="cpu")
                underlying_model = cls.auto_model_loader.from_pretrained(config.underlying_config, *model_args,
                                                                         **kwargs)
                model = cls(underlying_model, config)
                cls._load_state_dict(backbone_state_dict, model, underlying_model)
                return model

        else:
            # assume it is a model card on the hub                                                         )
            underlying_model = cls.auto_model_loader.from_pretrained(config.underlying_config, *model_args, **kwargs)

            state_dict = cls._load_remote_state_dict(kwargs, pretrained_model_name_or_path, state_dict)

        sled_model = cls(underlying_model, config)
        if state_dict is not None:
            logger.info('Updating SLED model weights from state dict')
            cls._load_state_dict(state_dict, sled_model)
        return sled_model

    @classmethod
    def _load_state_dict(cls, backbone_state_dict, model, underlying_model=None):
        load_result = model.load_state_dict(backbone_state_dict, strict=False)
        # Known issue - when loading a model checkpoint of type AutoModelForSeq2SeqLM with AutoModel,
        # the state dict will not be loaded correctly. Documented [here](https://github.com/Mivg/SLED/issues/4)
        if len(load_result.missing_keys) != 0:
            if underlying_model is not None and \
                    model._keys_to_ignore_on_save is not None and \
                    set(load_result.missing_keys) == set(model._keys_to_ignore_on_save):
                underlying_model.tie_weights()
            else:
                logger.warn(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warn(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

    @classmethod
    def _load_remote_state_dict(cls, kwargs, pretrained_model_name_or_path, state_dict):
        if state_dict is None:
            state_dict = _download_weights(kwargs, pretrained_model_name_or_path)
        return state_dict

    def _verify_config_consistency(self):
        if not self._verified_config_consistency:
            # SLED models are built in one of the following ways:
            # 1. Explicitly (given a SledConfig and an underlying model)
            # 2. from_pretrained (local_config, hub config, saved checkpoint)
            # There are 2 cases where the underlying_config and the SledConfig may mismatch - 1, and 2.saved_checkpoint
            # Instead of deciding whether to update the underlying model's config or vice versa while we cannot know which
            # one is correct, it is better to raise an exception. The only key we were willing to tolerate is the
            # vocab size, and we dealt with it above
            # Note - we will only do it once on the first forward pass. Setting a config in the model after it has
            # been used is non advisble
            config_dict = self.config.to_dict()
            underlying_config_dict = self.underlying_model.config.to_dict()
            matching_keys = set(config_dict.keys()).intersection(underlying_config_dict.keys()).difference(self._ignore_keys)
            inconsistent_keys = {k: (config_dict[k], underlying_config_dict[k])
                                 for k in matching_keys if config_dict[k] != underlying_config_dict[k]}

            if len(inconsistent_keys) > 0:
                # raise ValueError(f'SledConfig and the underlying_model config has mismatching configurations on: {inconsistent_keys}')
                # if we loaded the config with overrides, there may still be some conflicts
                logger.warning(
                    f'SledConfig and the underlying_model config has mismatching configurations on: {inconsistent_keys}, '
                    f'probably due to config_overrides. Setting the underlying config to match SLEDs')
                for k in inconsistent_keys:
                    setattr(self.underlying_model.config, k, getattr(self.config, k))

            config = self.config
            if  self._window_fraction != config.window_fraction or \
                    self._context_size != config.context_size or\
                    self._window_margin != int(config.context_size * (1 - config.window_fraction) / 2) or \
                    self._sliding_method != config.sliding_method:
                raise RuntimeError('SLED does not support changing its configuration after it is initialized. '
                                   'Try reloading the model with overrides')

            self._verified_config_consistency = True

    @property
    def underlying_model(self):
        return self._underlying_model

    def resize_token_embeddings(self, new_num_tokens=None):
        res = self.underlying_model.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = self.underlying_model.vocab_size  # sync them
        return res

    @property
    def _ignore_keys(self):
        return self.IGNORE_CONFIG_KEYS

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica.forward = create_function(self._signature, replica._forward)
        return replica

    def _create_forward_function(self):
        # https://stackoverflow.com/a/15638720
        self._underlying_model_signature = inspect.signature(self._underlying_model.forward)
        self._forward_kwargs_names = [param.name for param in self._underlying_model_signature.parameters.values()]
        assert PREFIX_KEY not in self._forward_kwargs_names
        # if we want to prepend questions in every window, we need to set the forward signature to expect the
        # input_prefix (e.g. question) as a separate input sequence

        # we want to remove any typing information as it may cause issues in the custom function build do to
        # non-imported modules. It is ugly and shouldn't be done like that, but it works..
        params = [self._underlying_model_signature.parameters[p].replace(annotation=inspect.Parameter.empty)
                  for p in self._underlying_model_signature.parameters]
        params.append(inspect.Parameter(name=PREFIX_KEY, default=None, kind=params[-1].kind))
        self._signature = str(self._underlying_model_signature.replace(parameters=params,
                                                                       return_annotation=inspect.Signature.empty))

        # HF trainer uses the signature to choose which parts to take from a dataset, so we need to make sure our
        # wrapped forward function has the correct signature (dynamically creating it here)
        self.forward = create_function(self._signature, self._forward)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            try:
                return self._underlying_model.__getattribute__(item)
            except AttributeError:
                return self._underlying_model.__getattr__(item)

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        # the actual forward implementation of the model.
        raise NotImplementedError

    def _set_underlying_model_attr(self, attr_name, new_val):
        if hasattr(self._underlying_model, attr_name):
            setattr(self._underlying_model, attr_name, new_val)
        elif hasattr(self._underlying_model, "model") and hasattr(self._underlying_model.model, attr_name):
            setattr(self._underlying_model.model, attr_name, new_val)
        else:
            raise ValueError(f"Cannot use this model as we cannot set its {attr_name}")

    def _run_sliding_window_forward(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                    prefix_length=None, **kwargs):
        sm = self._sliding_method if self._sliding_method != 'dynamic' else \
            ('loop' if not self.training else 'stacked')
        try:
            if sm == 'decoderonly':
                return self._skip_forward_for_decoder_only(args_tensor_inds, kwargs_tensor_keys, s, *args,
                                                           prefix_length=prefix_length, **kwargs)
            if sm == 'loop':
                return self._run_sliding_window_forward_loop(args_tensor_inds, kwargs_tensor_keys, s, *args,
                                        prefix_length=prefix_length, **kwargs)
            return self._run_sliding_window_forward_stacked(args_tensor_inds, kwargs_tensor_keys, s, *args,
                                                         prefix_length=prefix_length, **kwargs)
        finally:
            # so that if the model crashes halfway through it will be restored to working order
            pass

    def _skip_forward_for_decoder_only(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                       prefix_length=None, **kwargs):
        # NOTE - this will work probably only with BART.
        embeder = self if hasattr(self, 'embed_tokens') else self.get_encoder() # account for sled encoder
        return (embeder.embed_tokens(kwargs['input_ids']), )


    def _run_sliding_window_forward_loop(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                    prefix_length=None, **kwargs):
        forward_kwargs = _extract_keyword_args(kwargs, self._forward_kwargs_names, None)
        encoder_outputs_list = []
        if prefix_length is not None and self._prepend_prefix:
            # we were given prefixes in the input, and we are expected to treat them
            prefix_length, s = self._handle_prefix(prefix_length, s)

            if self._encode_prefix:
                # encode the question as well, if needed
                context_start_ind, context_end_ind, update_start_ind, update_end_ind = 0, prefix_length, 0, prefix_length

                encoder_outputs = self._underlying_model.forward(
                    *_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, None),
                    **_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind, context_end_ind, None),
                )
                encoder_outputs_list.append((encoder_outputs, update_start_ind, update_end_ind, None))
                # we will need to make sure all input tensors will also drop everything with the prefix
        else:
            prefix_length = None  # we need to ignore the prefix and treat the entire input as one long document

        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in self._window_indices(s):
            encoder_outputs = self._underlying_model.forward(
                *_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, prefix_length),
                **_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind, context_end_ind, prefix_length),
            )
            encoder_outputs_list.append((encoder_outputs, update_start_ind, update_end_ind, prefix_length))

        return _merge_encoder_outputs(encoder_outputs_list)

    def _handle_prefix(self, prefix_length, s):
        prefix_length_ = prefix_length[0].detach().cpu().item()
        assert torch.all(prefix_length == prefix_length_).item(), \
            'Using different length prefixes in the same batch is not supported. Either group your batch by ' \
            'prefix length, or pad the prefixes to match in length (and do not forget to set the attention ' \
            'mask to 0 where appropriate)'
        if hasattr(self.underlying_model.config, 'max_position_embeddings'):
            assert self._context_size + prefix_length_ <= self.underlying_model.config.max_position_embeddings, \
                f'The prefix length + SLEDs chunk size must be at most the max length that the backbone model can handle'
        return prefix_length_, s-prefix_length_

    def _run_sliding_window_forward_stacked(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                    prefix_length=None, **kwargs):
        forward_kwargs = _extract_keyword_args(kwargs, self._forward_kwargs_names, None)
        stacks_args = []
        stacks_kwargs = []
        stacks_info = []

        if prefix_length is not None and self._prepend_prefix:
            # we were given prefixes in the input, and we are expected to treat them
            prefix_length, s = self._handle_prefix(prefix_length, s)

            if self._encode_prefix:
                # encode the question as well, if needed
                context_start_ind, context_end_ind, update_start_ind, update_end_ind = 0, prefix_length, 0, prefix_length
                # need to pad it to match the seq len of the rest
                # we may have too short samples as well so don't want to pad too much
                pad = min(s, self._context_size)
                assert pad >= 0, f'We have a weird situation. pad={pad}, s={s}, ' \
                                 f'prefix_length={prefix_length} and self._context_size={self._context_size}'
                stacks_args.append(_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, None, pad))
                stacks_kwargs.append(_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind,
                                                 context_end_ind, None, pad))
                stacks_info.append([None, update_start_ind, update_end_ind, None])
        else:
            prefix_length = None  # we need to ignore the prefix and treat the entire input as one long document

        for context_start_ind, context_end_ind, update_start_ind, update_end_ind in self._window_indices(s):
            stacks_args.append(_fix_args(args, args_tensor_inds, context_start_ind, context_end_ind, prefix_length))
            stacks_kwargs.append(_fix_kwargs(forward_kwargs, kwargs_tensor_keys, context_start_ind,
                                             context_end_ind, prefix_length))
            stacks_info.append([None, update_start_ind, update_end_ind, prefix_length])

        encoder_outputs2 = self._underlying_model.forward(
            *_stack_args(stacks_args, args_tensor_inds),
            **_stack_kwargs(stacks_kwargs, kwargs_tensor_keys))
        bs = forward_kwargs[kwargs_tensor_keys[0]].size()[0] if len(kwargs_tensor_keys) > 0 else \
            args[args_tensor_inds[0]].size()[0]
        for si, eo in zip(stacks_info, _unstack_encoder_outputs(encoder_outputs2, len(stacks_info), bs)):
            si[0] = eo
        res = _merge_encoder_outputs(stacks_info)

        return res

    def _window_indices(self, total_seq_len):
        """
        when total_seq_len is smaller than our desired context length, we do not do sliding window at all.
        However, if it is longer, then we ALWAYS require the context length to be maximal, even if some windows have
        a lot of overlap.
        Also, first window will always update from the start, and last window will always update until the end.
        when applied, returns a generator that in each iteration produces for numbers:
        context_start_ind, context_end_ind, update_start_ind, update_end_ind

        context_start_ind, context_end_ind are indices in [0, total_seq_len],
        where context_end_ind > context_start_ind and when
        total_seq_len <= context_length then always context_end_ind = context_start_ind+context_length.
        The sequence of context_start_ind is strictly monotonic and same for context_end_ind.
        context_start_ind always start in 0 and
        context_end_ind will always end in total_seq_len.
        Gives us what token indices to take from the long input.

        update_start_ind, update_end_ind are indices in [0, min(total_seq_len, context_length)],
        where update_end_ind > update_start_ind
        and for all windows that are not in the edges (i.e. first/last window) we have
        update_end_ind-update_start_ind=context_length*window_fraction.
        For first window update_start_ind is always 0, and for last window,
        update_end_ind is always min(total_seq_len, context_length).
        They represents the start and end indices from the selected window of
        which tokens should be taken out for the final encoding

        When doing a full itartion, accounting for the fact that
        update_start_ind, update_end_ind are shifted by context_start_ind, we hould get that all indices in
        [0, total_seq_len] were covered exactly once

        Examples
        >>> from transformers import T5Tokenizer, T5Model
        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model_ = T5Model.from_pretrained('t5-small')
        >>> model = SledModel(model_, 512)  # testing with padding of 50% and context of 512

        >>> list(model._window_indices(256))  # List of: (context_start, context_end, update_start, update_end). short sequence
        [(0, 256, 0, 256)]
        >>> list(model._window_indices(510))  # another short sequence
        [(0, 510, 0, 510)]
        >>> list(model._window_indices(512))  # sequence of exactly the context size
        [(0, 512, 0, 512)]
        >>> list(model._window_indices(514))  # sequence of slightly more than the context size
        [(0, 512, 0, 384), (2, 514, 382, 512)]
        >>> list(model._window_indices(766))  # long sequence that does not require a full stride (update in the last chunk is smaller than what is possible)
        [(0, 512, 0, 384), (254, 766, 130, 512)]
        >>> list(model._window_indices(768))  # long sequence for exactly two perfect chunks
        [(0, 512, 0, 384), (256, 768, 128, 512)]
        >>> list(model._window_indices(780))  # very long sequence that does not require a full stride (update in the last chunk is smaller than what is possible)
        [(0, 512, 0, 384), (256, 768, 128, 384), (268, 780, 372, 512)]
        >>> windows = list(model._window_indices(1050))
        >>> windows
        [(0, 512, 0, 384), (256, 768, 128, 384), (512, 1024, 128, 384), (538, 1050, 358, 512)]
        >>> windows = sum([list(range(us+cs, ue+cs)) for cs, _, us, ue in windows], [])  # verify it covers exactly all the indices, each once
        >>> windows[:10]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> windows[500:510]
        [500, 501, 502, 503, 504, 505, 506, 507, 508, 509]
        >>> len(windows)
        1050
        >>> len(set(windows))
        1050
        >>> model = SledModel(model_, 256, window_fraction=0.75)  # now testing with padding of 25% and context of 256

        >>> list(model._window_indices(128))  # List of: (context_start, context_end, update_start, update_end). short sequence
        [(0, 128, 0, 128)]
        >>> list(model._window_indices(254))  # another short sequence
        [(0, 254, 0, 254)]
        >>> list(model._window_indices(256))  # sequence of exactly the context size
        [(0, 256, 0, 256)]
        >>> list(model._window_indices(258))  # sequence of slightly more than the context size. margin is 256/8 -> 32
        [(0, 256, 0, 224), (2, 258, 222, 256)]
        >>> list(model._window_indices(446))  # long sequence that does not require a full stride (update in the last chunk is smaller than what is possible). stride should be 256-64=192
        [(0, 256, 0, 224), (190, 446, 34, 256)]
        >>> list(model._window_indices(448))  # long sequence for exactly two perfect chunks
        [(0, 256, 0, 224), (192, 448, 32, 256)]
        >>> list(model._window_indices(500))  # very long sequence that does not require a full stride (update in the last chunk is smaller than what is possible)
        [(0, 256, 0, 224), (192, 448, 32, 224), (244, 500, 172, 256)]
        >>> windows = list(model._window_indices(1050))
        >>> windows
        [(0, 256, 0, 224), (192, 448, 32, 224), (384, 640, 32, 224), (576, 832, 32, 224), (768, 1024, 32, 224), (794, 1050, 198, 256)]
        >>> windows = sum([list(range(us+cs, ue+cs)) for cs, _, us, ue in windows], [])  # verify it covers exactly all the indices, each once
        >>> windows[:10]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> windows[500:510]
        [500, 501, 502, 503, 504, 505, 506, 507, 508, 509]
        >>> len(windows)
        1050
        >>> len(set(windows))
        1050
        """
        if total_seq_len <= self._context_size:
            yield 0, total_seq_len, 0, total_seq_len
        else:
            stride = self._context_size - 2 * self._window_margin
            context_start = update_start_ind = 0
            context_end = self._context_size
            update_end_ind = context_end - self._window_margin
            yield context_start, context_end, update_start_ind, update_end_ind  # first window always should update from the beginning
            while context_end < total_seq_len:
                context_end = min(total_seq_len, context_end + stride)
                context_start = (
                    context_start + stride if context_end < total_seq_len else total_seq_len - self._context_size
                )
                update_start_ind = max(update_start_ind + stride, update_end_ind)
                # last window always should update until the end
                update_end_ind = (
                    min(total_seq_len, update_end_ind + stride) if context_end < total_seq_len else total_seq_len
                )

                cs, ce, us, ue = context_start, context_end, update_start_ind - context_start, \
                                 update_end_ind - context_start

                yield cs, ce, us, ue

    def _fill_prefix_inputs(self, kwargs, kwargs_tensor_keys):
        prefix_inputs = {}
        k = PREFIX_KEY
        if PREFIX_KEY in kwargs:
            if self._prepend_prefix:
                if k not in kwargs_tensor_keys:
                    warnings.warn(f'{k} is missing from kwargs_tensor_keys (though expected for SLED prefix prepending)')
                else:
                    kwargs_tensor_keys.remove(k)
                    prefix_inputs[k] = kwargs.pop(k)
            elif k in kwargs_tensor_keys:
                warnings.warn(f'{k} is given in kwargs_tensor_keys even though sled should not prepend prefix, '
                              f'that would mean the prefix would be ignored and the entire input will be treated '
                              f'as a single long document, which is probably not what you meant')
        return prefix_inputs

    @staticmethod
    def _prep_attention_mask_for_cross_attention(encode_prefix, attention_mask, prefix_length=None):
        # if we need to drop the prefix encodings, we also need to adjust the attention mask before decoding
        if not encode_prefix and prefix_length is not None:
            prefix_length = int(prefix_length[0])
            return attention_mask[..., prefix_length:]
        return attention_mask


class SledModel(SledPretrainedModel):
    """
    >>> from transformers import T5Tokenizer, T5Model, BartModel, BartTokenizer

    >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
    >>> model_ = T5Model.from_pretrained('t5-small')
    >>> model = SledModel(model_, 4)

    >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
    >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
    >>> outputs = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone())
    >>> outputs = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=True)
    >>> outputs = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=False)
    """

    def __init__(self, underlying_model: PreTrainedModel, config: SledConfig):
        super(SledModel, self).__init__(underlying_model, config)
        # validate the model can be used
        self._decoder_attr_name = getattr(underlying_model, "get_decoder_attr_name", lambda: "decoder")()
        self._encoder_attr_name = getattr(underlying_model, "get_encoder_attr_name", lambda: "encoder")()
        self._set_underlying_model_attr(self._decoder_attr_name, self.get_decoder())
        self._mock_decoder = MockDecoder()
        assert "return_dict" in self._forward_kwargs_names
        assert "encoder_outputs" in self._forward_kwargs_names

    def _forward(self, *args, **kwargs):
        self._verify_config_consistency()
        kwargs, args = _fill_kwargs_with_args(self._forward_kwargs_names, *args, **kwargs)
        kwargs.setdefault("encoder_outputs", None)
        return_dict = kwargs.setdefault("return_dict", None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        labels = kwargs.get("labels", None)
        kwargs["labels"] = None
        kwargs["return_dict"] = False
        kwargs.setdefault("labels", None)
        args_tensor_inds, kwargs_tensor_keys, s = _find_tensor_inds_and_size(*args, **kwargs)
        prefix_inputs = self._fill_prefix_inputs(kwargs, kwargs_tensor_keys)

        forward_kwargs = _extract_keyword_args(kwargs, self._forward_kwargs_names, None)
        if forward_kwargs["encoder_outputs"] is None:
            # encode, but first let's set decoder to be a mock, no reason to apply it over partial windows
            self._prep_for_encoding()  # todo - add try catch every time we 'prep' something to rever the state on fail?

            forward_kwargs["encoder_outputs"] = self._run_sliding_window_forward(
                args_tensor_inds, kwargs_tensor_keys, s, *args, **prefix_inputs, **forward_kwargs
            )
            forward_kwargs['attention_mask'] = self._prep_attention_mask_for_cross_attention(self._encode_prefix,
                forward_kwargs['attention_mask'], prefix_inputs.get('prefix_length', None))

        # now, let's decode
        forward_kwargs["return_dict"] = return_dict
        forward_kwargs["labels"] = labels
        self._fix_post_encoding()
        if 'decoder_input_ids' in self._forward_kwargs_names and \
            forward_kwargs.get('decoder_input_ids', None) is None and \
                hasattr(self, 'prepare_decoder_input_ids_from_labels') :
            logger.warning('Passing a batch through the model without the decoder_input_ids is likely to cause issues. '
                           'If you encounter cuda errors, make sure you use the prepare_decoder_input_ids_from_labels '
                           'function of the model correctly before passing the input. '
                           'If you are only performing prediction without training, you can safely ignore this message')
        res = self._underlying_model.forward(
            *args, **_extract_keyword_args(forward_kwargs, self._forward_kwargs_names)
        )

        return res

    def _prep_for_encoding(self):
        if not getattr(self, '_preped_for_encoding', False):
            self._preped_for_encoding = True
            self._decoder = self.get_decoder()
            self._mock_decoder.first_device = getattr(self._decoder, "first_device", None)
            self._set_underlying_model_attr(self._decoder_attr_name, self._mock_decoder)

    def _fix_post_encoding(self):
        assert self._preped_for_encoding
        self._preped_for_encoding = False
        self._set_underlying_model_attr(self._decoder_attr_name, self._decoder)


class MockTensorForConditionalGeneration:
    def __add__(self, other):
        return tuple()

    def __mul__(self, other):
        return tuple()

    def to(self, *_, **__):
        return self


class MockDecoderForConditionalGeneration(nn.Module):
    pad_value = 0

    def forward(self, *_, **__):
        return (MockTensorForConditionalGeneration(),)

    def to(self, *_, **__):
        return self


class MockLMHeadForConditionalGeneration(nn.Module):
    def forward(self, *_, **__):
        return MockTensorForConditionalGeneration()

    def to(self, *_, **__):
        return self

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return self


def _fill_kwargs_with_args(forward_param_names, *args, **kwargs):
    kwargs.update({arg_name: arg_value for arg_name, arg_value in zip(forward_param_names, args)})
    return kwargs, tuple()


class SledEncoder(SledPretrainedModel):
    def __init__(self, underlying_model: PreTrainedModel, config: SledConfig):
        super(SledEncoder, self).__init__(underlying_model, config)

    @property
    def _ignore_keys(self):
        return super()._ignore_keys | {'use_cache', 'is_encoder_decoder'}
    # Encoder models are not encoder-decoder but are meant for internal use by the SLED models


    def _forward(self, *args, **kwargs):
        kwargs, args = _fill_kwargs_with_args(self._forward_kwargs_names, *args, **kwargs)
        args_tensor_inds, kwargs_tensor_keys, s = _find_tensor_inds_and_size(*args, **kwargs)
        prefix_inputs = self._fill_prefix_inputs(kwargs, kwargs_tensor_keys)
        return self._run_sliding_window_forward(args_tensor_inds, kwargs_tensor_keys, s, *args, **prefix_inputs,
                                                **kwargs)

    def _skip_forward_for_decoder_only(self, args_tensor_inds, kwargs_tensor_keys, s, *args,
                                                           prefix_length=None, **kwargs):
        encoder_outputs = super()._skip_forward_for_decoder_only(args_tensor_inds, kwargs_tensor_keys, s,
                                                                           *args, prefix_length, **kwargs)
        return BaseModelOutput(encoder_outputs[0])


class SledForConditionalGeneration(SledModel):
    """
    >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

    >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
    >>> model_ = T5ForConditionalGeneration.from_pretrained('t5-small')
    >>> shared = model_.shared
    >>> model = SledForConditionalGeneration(model_, 4)
    >>> model._underlying_model == model_  # make sure the decoration works
    >>> model.shared == shared  # make sure the decoration works
    >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
    >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
    >>> outputs = model(input_ids=input_ids, labels=labels)
    >>> outputs = model(input_ids=input_ids, labels=labels, return_dict=None)
    >>> outputs = model(input_ids=input_ids, labels=labels, return_dict=False)
    >>> outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
    >>> outputs = model.generate(input_ids)
    """
    OVERRIDDEN_METHODS = {'generate'}
    auto_model_loader = AutoModelForSeq2SeqLM

    def __init__(self, underlying_model: PreTrainedModel, config: SledConfig):
        super(SledForConditionalGeneration, self).__init__(underlying_model, config)
        self._mock_decoder = MockDecoderForConditionalGeneration()
        self._base_encoder = self.get_encoder()
        self._sled_encoder = SledEncoder(self._base_encoder, config)

        # override generation preparation functions that may be overridden by underlying models but will be found in our wrapper
        for method_name, _ in inspect.getmembers(GenerationMixin(), predicate=inspect.ismethod):
            if method_name not in SledForConditionalGeneration.OVERRIDDEN_METHODS:
                setattr(self, method_name, getattr(underlying_model, method_name))

        # NOTE - the below affects the given underlying model, which means generating with it
        # directly may not work anymore
        self._underlying_model._prepare_encoder_decoder_kwargs_for_generation = \
            self._get__prepare_encoder_decoder_kwargs_for_generation_func_override()
        self._underlying_model._validate_model_kwargs = _validate_model_kwargs  # see hack details below

    def _get__prepare_encoder_decoder_kwargs_for_generation_func_override(self):
        # _prepare_encoder_decoder_kwargs_for_generation(
        #         self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
        #     ) -> Dict[str, Any]:
        f = self._underlying_model._prepare_encoder_decoder_kwargs_for_generation
        encode_prefix = self._encode_prefix

        def _prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
            # override needed for underlying model in a conditional generation mode with prefix prepending and dropping
            model_kwargs = f(inputs_tensor, model_kwargs, model_input_name)
            model_kwargs['attention_mask'] = SledPretrainedModel._prep_attention_mask_for_cross_attention(
                encode_prefix, model_kwargs['attention_mask'], model_kwargs.get('prefix_length', None))
            return model_kwargs

        return _prepare_encoder_decoder_kwargs_for_generation

    def _prep_for_encoding(self):
        was_preped = getattr(self, '_preped_for_encoding', False)
        super(SledForConditionalGeneration, self)._prep_for_encoding()
        if not was_preped:
            self._lm_head = getattr(self._underlying_model, "lm_head", None)
            setattr(self._underlying_model, "lm_head", MockLMHeadForConditionalGeneration())

    def _fix_post_encoding(self):
        super(SledForConditionalGeneration, self)._fix_post_encoding()
        setattr(self._underlying_model, "lm_head", self._lm_head)

    def generate(self, *args, **kwargs):
        self._set_underlying_model_attr(self._encoder_attr_name, self._sled_encoder)
        try:
            res = self._underlying_model.generate(*args, **kwargs)
        finally:
            self._set_underlying_model_attr(self._encoder_attr_name, self._base_encoder)
        return res

def _validate_model_kwargs(self, *args, **kwargs):
    # Newer versions of HF perform a check on the input args for generate and raise an exception when passing
    # prefix_length for example to this model because it doesn't list it explicitly.
    # This is a hack to support newer HF models until the generate() signature will be created dynamically to
    # include all the keyword args including prefix_length
    pass
