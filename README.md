# SLED
The official repository for Efficient Long-Text Understanding Using Short-Text Models [(Ivgi et al., 2022)](https://arxiv.org/abs/2208.00748.pdf).

SLED models use pretrained, short-range encoder-decoder models, and apply them over. 
long-text inputs by splitting the input into multiple overlapping chunks, encoding each independently and perform fusion-in-decoder.


## Data
The data for this paper is hosted on the dataset hub [here](https://huggingface.co/datasets/tau/sled). 
It is based on the [SCROLLS dataset](https://huggingface.co/datasets/tau/scrolls) ([paper](https://arxiv.org/pdf/2201.03533.pdf)), the [SQuAD 1.1 dataset](https://huggingface.co/datasets/squad) ([paper](https://arxiv.org/pdf/1606.05250.pdf)) and the [HotpotQA dataset](https://huggingface.co/datasets/hotpot_qa) ([paper](https://arxiv.org/pdf/1809.09600.pdf)).
It doesn't contain any unpublished data, but includes the configuration needed for the paper.

Usage example :
```python
from datasets import load_dataset
qasper = load_dataset("tau/sled","qasper")
```

## Installation

Make sure to install pytorch according to your machine spec. See installation options [here](https://pytorch.org/get-started/locally/).

Installing SLED is easy with pip.
```
pip install py-sled
```

Some backbone models require additional dependencies. If you wish to work with T5 for example, you can install using.
```
pip install py-sled[t5]
```

If you wish to run the examples, install the required dependencies with
```
pip install py-sled[examples]
```

If you wish to continue developing this repository, install the full development requirments with
```
pip install py-sled[dev]
```

## Usage
Working with SLED is seamless when working with HuggingFace's Transformers AutoClasses.

A minimal usage example:
```python
import sled  # ** required so SLED would be properly registered by the AutoClasses **
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('tau/bart-base-sled')
model = AutoModel.from_pretrained('tau/bart-base-sled')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

_Important_: You need to `import sled` before using the AutoClass (e.g. `AutoModel.from_pretrained('tau/bart-base-sled)`) for it to work.

Minimal working example can be found [here](examples/usage_example.py).

To work with SCROLLS like data that was used for the paper, see [here](examples/seq2seq).

### Custom datasets
For SLED to be able to prepend the prefix input to every chunk, it requires the input tensor `prefix_length`. 
If using a custom dataset, you can refer to [run.py](examples/seq2seq/run.py) for the correct way to preprocess the data.

_Note_: Currently, HF's Seq2SeqTrainer doesn't pass the `prefix_length` tensor in the prediction loop, so you 
 should use the [CustomSeq2SeqTrainer](examples/seq2seq/utils/custom_seq2seq_trainer.py) or something similar until it is 
fixed.

### Backbone models
There are multiple model cards available on HuggingfaceHub including
- [Bart-Base SLED](https://huggingface.co/tau/bart-base-sled) (model name `tau/bart-base-sled`)
- [Bart-Large SLED](https://huggingface.co/tau/bart-large-sled) (model name `tau/bart-base-sled`)
- [T5(v1.1)-base SLED](https://huggingface.co/tau/t5-v1_1-base-sled) (model name `tau/t5-v1_1-base-sled`)
- [T5(v1.1)-large SLED](https://huggingface.co/tau/t5-v1_1-large-sled) (model name `tau/t5-v1_1-large-sled`)

If you wish to use a custom model that is available as a model card (public or private) on the hub, or use 
different parameters for SLED, you can create a json config file like the below, and change the underlying_config to your custom model card.
```json
{
  "model_type": "tau/sled",
  "underlying_config": "facebook/bart-base",
  "context_size": 256,
  "window_fraction": 0.5,
  "prepend_prefix": true,
  "encode_prefix": true,
  "sliding_method": "dynamic"
}
```
You can then load it like below
```python
import sled
from transformers import AutoModelForSeq2SeqLM
custom_sled_model = AutoModelForSeq2SeqLM.from_pretrained(<your custom json config>)
```

## Citation

If you use this repository, please cite as below:
```
@inproceedings{Ivgi2022EfficientLU,
  title={Efficient Long-Text Understanding with Short-Text Models},
  author={Maor Ivgi and Uri Shaham and Jonathan Berant},
  year={2022}
}
```


## Disclaimer
This repository is still under active development, and may contain some unintended behavior. 
Please open an issue if any unexpected behaviour occurs, and we will promptly try to fix it.
