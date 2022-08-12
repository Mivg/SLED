import os

from transformers import PreTrainedTokenizer, AutoTokenizer, AutoConfig


class SledTokenizer(PreTrainedTokenizer):
    auto_tokenizer_loader = AutoTokenizer
    auto_config_loader = AutoConfig

    def __init__(self, *args, **kwargs):
        super(SledTokenizer, self).__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        assert isinstance(pretrained_model_name_or_path, str), 'pretrained_model_name_or_path must be a path to a ' \
                                                               'checkpoint or a local config file (json)'
        if os.path.exists(pretrained_model_name_or_path):
            # if pretrained_model_name_or_path is a saved checkpoint
            config = kwargs.pop('config', None)
            if pretrained_model_name_or_path.endswith('json'):
                config = config or cls.auto_config_loader.from_pretrained(pretrained_model_name_or_path)
                return cls.auto_tokenizer_loader.from_pretrained(config.underlying_config, *init_inputs, **kwargs)
            else:
                # otherwise, it is a config json path
                raise NotImplementedError('loading a pretrained saved sled checkpoint is not yet implemented')
        else:
            # assume it is a model card on the hub
            config = kwargs.pop('config', None)
            config = config or cls.auto_config_loader.from_pretrained(
                pretrained_model_name_or_path, use_auth_token=kwargs.get('use_auth_token', False))
            kwargs['use_fast'] = False
            return cls.auto_tokenizer_loader.from_pretrained(config.underlying_config, *init_inputs, **kwargs)