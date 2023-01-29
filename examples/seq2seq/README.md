# Seq2Seq finetuning
In this example, you can find the scripts needed to finetune sled models on SCROLLS like data.

The entrypoint script is [run.py](run.py)

## Usage
After setting up your environment as described [here](https://github.com/Mivg/SLED#installation), you can run the script to finetune, 
evaluate and generate predictions for all the datasets in the [SLED dataset](https://huggingface.co/datasets/tau/sled) 
(based on SCROLLS).

Like most recipes, you can view all possible settable parameters by running 
```
python run.py --help
```

To run, you can either set the parameters with command-line arguments (e.g. `--model_name_or_path tau/bart-base-sled`) 
or use  some predfined json files to set recurring configurations. You can pass as many json files as you would like, 
but make sure you pass them before any other command line argument. For example, you can do the following:
```
python run.py configs/data/squad.json \
configs/model/bart_base_sled.json \
configs/training/base_training_args.json \
--output_dir /tmp/output_sled
--learning_rate 2e-5
```

Example jsons files are [here](https://github.com/Mivg/SLED/tree/main/examples/seq2seq/configs).
