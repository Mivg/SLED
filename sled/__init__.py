__version__ = '0.1.7'

try:
    # noinspection PyPackageRequirements
    import torch
except ImportError:
    raise ImportError('Using sled requires torch. Please refer to https://pytorch.org/get-started/locally/ '
                      'to install the correct version for your setup')

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

from .configuration_sled import SledConfig
# noinspection PyUnresolvedReferences
from .modeling_sled import SledModel, SledForConditionalGeneration, PREFIX_KEY
from .tokenization_sled import SledTokenizer
from .tokenization_sled_fast import SledTokenizerFast

AutoConfig.register('tau/sled', SledConfig)
AutoModel.register(SledConfig, SledModel)
AutoModelForSeq2SeqLM.register(SledConfig, SledForConditionalGeneration)
AutoTokenizer.register(SledConfig, slow_tokenizer_class=SledTokenizer, fast_tokenizer_class=SledTokenizerFast)