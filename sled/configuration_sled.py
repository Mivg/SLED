from typing import Dict, Any

from transformers import PretrainedConfig, AutoConfig


class SledConfig(PretrainedConfig):
    r"""
    """
    model_type = "tau/sled"

    def __init__(self, underlying_config="facebook/bart-base", context_size=256, window_fraction=0.5,
                 prepend_prefix=True, encode_prefix=True, sliding_method='dynamic', **kwargs):
        super().__init__(**kwargs)
        parent_only_keys = set(super().to_dict().keys())
        self.underlying_config = underlying_config
        self.context_size = context_size
        self.window_fraction = window_fraction
        self.prepend_prefix = prepend_prefix
        self.encode_prefix = encode_prefix
        self.sliding_method = sliding_method

        # load underlying_config
        config = AutoConfig.from_pretrained(underlying_config, **kwargs)
        # update internal dict based on the underlying config, overriding everything EXCEPT what was explicitly set here
        ignore_keys = set(self.to_dict().keys()).union(type(self).__dict__.keys()).difference(parent_only_keys)
        self.__dict__.update({k: v for k, v in config.to_dict().items() if k not in ignore_keys})
