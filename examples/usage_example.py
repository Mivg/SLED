"""
    Minimal working example to use an X-SLED model
"""
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
# noinspection PyUnresolvedReferences
import sled  # *** required so that SledModels will be registered for the AutoClasses ***

if __name__ == '__main__':
    # Load the model and tokenizer for Bart-base-SLED
    bart_base_sled_model = AutoModel.from_pretrained('tau/bart-base-sled')
    tokenizer = AutoTokenizer.from_pretrained('tau/bart-base-sled')
    bart_base_sled_model.eval()

    # The below example is for cases where there are no prefix (e.g. question) to use, such as summarization
    document_input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    )  # Batch size 1
    with torch.no_grad():
        final_representations = bart_base_sled_model(**document_input_ids, return_dict=None).last_hidden_state

    # Now, assume we do have a prefix (for example in question answering)
    prefix_input_ids = tokenizer(
        "Is owning a dog good for you?", return_tensors="pt"
    ).input_ids  # Batch size 1

    # we concatenate them together, but tell SLED where is the prefix by setting the prefix_length tensor
    input_ids = torch.cat((prefix_input_ids, document_input_ids.input_ids), dim=-1)
    attention_mask = torch.ones_like(input_ids)
    prefix_length = torch.LongTensor([[prefix_input_ids.size(1)]])
    with torch.no_grad():
        final_representations = bart_base_sled_model(input_ids=input_ids, attention_mask=attention_mask,
                                                     prefix_length=prefix_length, return_dict=None).last_hidden_state

    # However, we are dealing with a generative model here (encoder-decoder), so, we can use it to generate
    bart_base_sled_model = AutoModelForSeq2SeqLM.from_pretrained('tau/bart-base-sled')
    with torch.no_grad():
        generated_output = bart_base_sled_model.generate(input_ids=input_ids,
                                                         prefix_length=prefix_length, return_dict=None)
