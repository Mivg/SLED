import os
import shutil
import sys

from torch import nn

from sled.tokenization_sled import SledTokenizer
from sled.tokenization_sled_fast import SledTokenizerFast

BART_BASE_SLED_URL = 'tau/bart-base-sled'
BART_BASE_SLED_GOV_URL = 'tau/bart-base-sled-govreport'
T5_BASE_SLED_URL = 'tau/t5-v1_1-base-sled'

sys.path.insert(0, os.getcwd())

import inspect
import unittest

import torch
from transformers import (
    T5Tokenizer,
    T5Model,
    BartModel,
    BartTokenizer,
    T5ForConditionalGeneration,
    BartForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    BartTokenizerFast, T5TokenizerFast,
)
from transformers.testing_utils import require_torch


from sled.modeling_sled import SledModel, SledForConditionalGeneration, PREFIX_KEY
from sled.configuration_sled import SledConfig

use_auth_token = False

def _compare_tuple_of_tensors(test_case, expected_tuple, got_tuple, prev_got_tuple, rtol=1e-5):
    for i, (exp_tensor, got_tensor) in enumerate(zip(expected_tuple, got_tuple)):
        if isinstance(exp_tensor, torch.Tensor):
            test_case.assertTrue(torch.allclose(exp_tensor, got_tensor, rtol=rtol))
            # we can't expect the values to be the same when different context length, but at least can verify shapes
            if prev_got_tuple is not None:
                test_case.assertTrue(exp_tensor.size() == prev_got_tuple[i].size())
        elif isinstance(exp_tensor, tuple):
            prev_got_tuple_i = prev_got_tuple[i] if prev_got_tuple is not None else None
            _compare_tuple_of_tensors(test_case, exp_tensor, got_tensor, prev_got_tuple_i, rtol=rtol)


@require_torch
class SLEDModelTest(unittest.TestCase):
    def _run_sled_model_test_case(self, model_, tokenizer, underlying_config: str, rtol=1e-5):
        model = SledModel(model_, SledConfig(underlying_config, context_size=4))
        model.eval()  # only change the model to be in eval (inference) mode,
        # thus not changing layer_norm params and removing dropout

        input_ids = tokenizer(
            "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ).input_ids  # Batch size 1
        decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        # simple verification there are no failures in the flow itself
        with torch.no_grad():
            _ = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone())
            _ = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=None)
            outputs_dict = model(
                input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=True
            )
            outputs_no_dict = model(
                input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=False
            )

        # let's verify that if the sequence is short, we behave exactly as the base model
        model = SledModel(model_, SledConfig(underlying_config, context_size=512))
        with torch.no_grad():
            output_expected = model_(
                input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=False
            )

        # on non dict return type
        with torch.no_grad():
            output_got = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=False)
        self.assertEqual(type(output_expected), type(output_got))
        self.assertEqual(type(output_expected), type(outputs_no_dict))
        self.assertEqual(len(output_got), len(output_expected))  # should be tuple so it's ok
        self.assertEqual(len(output_got), len(outputs_no_dict))  # should be tuple so it's ok
        _compare_tuple_of_tensors(self, output_expected, output_got, outputs_no_dict, rtol=rtol)

        # on dict return type
        with torch.no_grad():
            output_expected = model_(
                input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=True
            )
            output_got = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=True)
        self.compare_outputs_dict(output_expected, output_got, outputs_dict, rtol=rtol)

    def compare_outputs_dict(self, output_expected, output_got, outputs_dict=None, rtol=1e-5):
        if output_got is None and outputs_dict is not None:
            output_got = output_expected
        if outputs_dict is None:
            outputs_dict = output_expected
        self.assertEqual(type(output_expected), type(output_got))
        self.assertEqual(type(output_expected), type(outputs_dict))
        self.assertListEqual(list(output_got.keys()), list(output_expected.keys()))
        self.assertListEqual(list(output_got.keys()), list(outputs_dict.keys()))
        for key in output_got.keys():
            if isinstance(output_got[key], torch.Tensor):
                self.assertTrue(torch.allclose(output_got[key], output_expected[key], rtol=rtol))
                self.assertFalse(output_got[key].requires_grad)
                self.assertFalse(output_expected[key].requires_grad)
                self.assertFalse(outputs_dict[key].requires_grad)

                # we can't expect the values to be the same when different context length, but at least can verify shapes
                self.assertTrue(output_got[key].size() == outputs_dict[key].size())
            elif isinstance(output_got[key], tuple):
                _compare_tuple_of_tensors(self, output_got[key], output_expected[key], outputs_dict[key], rtol=rtol)

    def test_facade_get_attr_behavior(self):
        base_model = T5Model.from_pretrained("t5-small")
        sled_model = SledModel(base_model, SledConfig("t5-small", context_size=4))
        self.assertEqual(
            base_model.shared, sled_model.shared
        )  # sled model does not have a 'shared' attribute, only it's base model does

    def _verify_loaded_sled_model(self, bart_sled_model, config_class, underlying_config_class,
                                  underlying_config_path="facebook/bart-base", expected_underlying_model=None):
        self.assertIsInstance(bart_sled_model, config_class)
        self.assertIsInstance(bart_sled_model._underlying_model, underlying_config_class)
        # make sure it has the pretrained weights and not random ones
        expected_underlying_model = expected_underlying_model or \
                                    underlying_config_class.from_pretrained(underlying_config_path)
        # noinspection PyTypeChecker
        self.assertTrue(torch.all(
            expected_underlying_model.get_encoder().state_dict()['embed_tokens.weight'] ==
            bart_sled_model.get_encoder().state_dict()['embed_tokens.weight']).item())

    def test_load_from_pretrained(self):
        config_path = BART_BASE_SLED_URL
        another_config_path = 'configs/t5_base_sled.json'
        another_config_path_hub = T5_BASE_SLED_URL

        # first, we test the config
        # start by loading it explicitly
        bart_base_sled_config = SledConfig.from_pretrained(config_path, use_auth_token=use_auth_token)
        self.assertIsInstance(bart_base_sled_config, SledConfig)

        # now, with auto classes
        bart_base_sled_config2 = AutoConfig.from_pretrained(config_path, use_auth_token=use_auth_token)
        self.assertNotEqual(bart_base_sled_config, bart_base_sled_config2) # the explicit one didn't have a name or path
        bart_base_sled_config._name_or_path = bart_base_sled_config2._name_or_path
        self.assertEqual(bart_base_sled_config, bart_base_sled_config2)

        # Now, let's check the model loading
        self._verify_loaded_sled_model(SledModel.from_pretrained(config_path, use_auth_token=use_auth_token),
                                       SledModel, BartModel)  # explicit load

        # now, with auto classes
        self._verify_loaded_sled_model(AutoModel.from_pretrained(config_path, use_auth_token=use_auth_token),
                                       SledModel, BartModel)  # auto load

        # now, lets assert that "forConditionalGeneration" Also works as expected
        self._verify_loaded_sled_model(SledForConditionalGeneration.from_pretrained(config_path,
                                                                                    use_auth_token=use_auth_token),
                                       SledForConditionalGeneration, BartForConditionalGeneration)  # explicit

        # now, with auto classes
        self._verify_loaded_sled_model(
            AutoModelForSeq2SeqLM.from_pretrained(config_path, use_auth_token=use_auth_token),
            SledForConditionalGeneration, BartForConditionalGeneration)  # auto

        # Finally, let's verify it also work with another model
        self._verify_loaded_sled_model(AutoModelForSeq2SeqLM.from_pretrained(
            another_config_path, use_auth_token=use_auth_token), SledForConditionalGeneration,
            T5ForConditionalGeneration, "google/t5-v1_1-base")

        self._verify_loaded_sled_model(AutoModelForSeq2SeqLM.from_pretrained(
            another_config_path_hub, use_auth_token=use_auth_token), SledForConditionalGeneration,
            T5ForConditionalGeneration, "google/t5-v1_1-base")

    def test_config_overrides(self):
        # Load the base model, and verify its consistency with the underlying config
        bart_base_model_sled = AutoModel.from_pretrained(BART_BASE_SLED_URL, use_auth_token=use_auth_token)
        self.assertFalse(bart_base_model_sled.config.gradient_checkpointing)
        self.assertFalse(bart_base_model_sled.underlying_model.config.gradient_checkpointing)
        self.assertTrue(bart_base_model_sled.config.use_cache)
        self.assertTrue(bart_base_model_sled.underlying_model.config.use_cache)

        # now, supply overrides and make sure they are consistent
        bart_base_model_sled = AutoModel.from_pretrained(BART_BASE_SLED_URL, gradient_checkpointing=True, use_cache=False,
                                                    use_auth_token=use_auth_token)
        self.assertTrue(bart_base_model_sled.config.gradient_checkpointing)
        self.assertTrue(bart_base_model_sled.underlying_model.config.gradient_checkpointing)
        self.assertFalse(bart_base_model_sled.config.use_cache)
        self.assertFalse(bart_base_model_sled.underlying_model.config.use_cache)

        # Finally, set the config after load and make sure it is synced properly
        bart_base_model_sled.config.gradient_checkpointing = False
        bart_base_model_sled.config.use_cache = True
        self.assertFalse(bart_base_model_sled.config.gradient_checkpointing)
        self.assertTrue(bart_base_model_sled.config.use_cache)
        # this wouldn't have been fixed before the model was first used
        self.assertTrue(bart_base_model_sled.underlying_model.config.gradient_checkpointing)
        self.assertFalse(bart_base_model_sled.underlying_model.config.use_cache)
        # now, pass something through the model (even though it would fail)
        try:
            bart_base_model_sled(None)
        except:
            pass
        self.assertFalse(bart_base_model_sled.underlying_model.config.gradient_checkpointing)
        self.assertTrue(bart_base_model_sled.underlying_model.config.use_cache)

    def test_all_loading_options(self):
        # test load from local config, from a saved checkpoint and from a URL model card
        local_bart_base_sled_model = SledModel.from_pretrained('configs/bart_base_sled.json')
        self._verify_loaded_sled_model(local_bart_base_sled_model, SledModel, BartModel)  # explicit load

        hub_bart_base_sled_model = SledModel.from_pretrained(BART_BASE_SLED_URL, use_auth_token=use_auth_token)
        self._verify_loaded_sled_model(hub_bart_base_sled_model, SledModel, BartModel)  # explicit load

        # now, save and reload
        cache_dir = os.environ.get('XDG_CACHE_HOME', '/tmp/cache')
        out_dir = f'{cache_dir}/test_save_checkpoint'
        os.makedirs(out_dir, exist_ok=True)
        shutil.rmtree(out_dir)  # cleanup previous checkpoints
        # let's change the model a little before saving it to make sure it works correctly and doesn't load the c
        # checkpoint on the hub
        local_bart_base_sled_model.get_encoder().state_dict()['embed_tokens.weight'] += 1
        # make sure they were indeed changed
        self.assertRaises(AssertionError, self._verify_loaded_sled_model, local_bart_base_sled_model, SledModel,
                          BartModel)

        # now save and test
        local_bart_base_sled_model.save_pretrained(out_dir)
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'pytorch_model.bin')))
        self.assertTrue(os.path.isfile(os.path.join(out_dir, 'config.json')))
        loaded_bart_base_sled_model = SledModel.from_pretrained(out_dir, use_auth_token=use_auth_token)
        self._verify_loaded_sled_model(loaded_bart_base_sled_model, SledModel, BartModel,
                                       expected_underlying_model=local_bart_base_sled_model)


    def test_load_tokenizer_from_pretrained(self):
        config_path = BART_BASE_SLED_URL
        another_config_path = T5_BASE_SLED_URL

        # slow tokenizer
        # explicit load, should actually return a BartTokenizer (the default is a fast tokenizer)
        self.assertIsInstance(SledTokenizer.from_pretrained(config_path, use_auth_token=use_auth_token, use_fast=False),
                              BartTokenizer)

        # autoload, should actually return a BartTokenizer
        self.assertIsInstance(AutoTokenizer.from_pretrained(config_path, use_auth_token=use_auth_token, use_fast=False),
                              BartTokenizer)

        # fast tokenizer
        # explicit load, should actually return a BartTokenizerFast
        self.assertIsInstance(SledTokenizerFast.from_pretrained(config_path, use_auth_token=use_auth_token),
                              BartTokenizerFast)
        # autoload, should actually return a BartTokenizerFast
        self.assertIsInstance(AutoTokenizer.from_pretrained(config_path, use_auth_token=use_auth_token),
                              BartTokenizerFast)

        # and now with T5
        self.assertIsInstance(SledTokenizer.from_pretrained(another_config_path, use_auth_token=use_auth_token,
                                                            use_fast=False), T5Tokenizer)
        self.assertIsInstance(AutoTokenizer.from_pretrained(another_config_path, use_auth_token=use_auth_token,
                                                            use_fast=False), T5Tokenizer)
        self.assertIsInstance(SledTokenizerFast.from_pretrained(another_config_path, use_auth_token=use_auth_token,
                                                            use_fast=True), T5TokenizerFast)
        self.assertIsInstance(AutoTokenizer.from_pretrained(another_config_path, use_auth_token=use_auth_token,
                                                            use_fast=True), T5TokenizerFast)

    def test_load_finetuned_model(self):
        bart_base_sled = AutoModel.from_pretrained(BART_BASE_SLED_URL)
        bart_base_sled_gov = SledModel.from_pretrained(BART_BASE_SLED_GOV_URL)
        # testing embeedings have changed
        assert not torch.all(bart_base_sled.get_input_embeddings().weight ==
                             bart_base_sled_gov.get_input_embeddings().weight).item()
        # test decoder weights have changed
        assert not torch.all(
            bart_base_sled.state_dict()['_underlying_model.decoder.layers.0.encoder_attn.k_proj.weight'] ==
            bart_base_sled_gov.state_dict()['_underlying_model.decoder.layers.0.encoder_attn.k_proj.weight']
        ).item()
        # test encoder weights have changed
        assert not torch.all(
            bart_base_sled.state_dict()['_underlying_model.encoder.layers.0.self_attn.k_proj.weight'] ==
            bart_base_sled_gov.state_dict()['_underlying_model.encoder.layers.0.self_attn.k_proj.weight']
        ).item()


    def test_sled_on_t5(self):
        self._run_sled_model_test_case(T5Model.from_pretrained("t5-small"), T5Tokenizer.from_pretrained("t5-small"),
                                       "t5-small")

    def test_sled_on_bart(self):
        self._run_sled_model_test_case(
            BartModel.from_pretrained("facebook/bart-base"), BartTokenizer.from_pretrained("facebook/bart-base"),
            "facebook/bart-base"
        )

    def test_forward_signature(self):
        # HF trainer uses the signature to choose which parts to take from a dataset, so we need to make sure our wrapped forward
        # function has the correct signature
        _model = BartModel.from_pretrained("facebook/bart-base")
        expected_sig = [param for param in inspect.signature(_model.forward).parameters.keys()]
        got_sig = [param for param in inspect.signature(SledModel(_model, SledConfig(context_size=128)).forward).parameters.keys()]
        self.assertListEqual(expected_sig, got_sig[:-1])
        self.assertEqual(str(got_sig[-1]), PREFIX_KEY)

    def test_resize_token_embeddings(self):
        _model = BartModel.from_pretrained("facebook/bart-base")
        orig_vocab_size = _model.config.vocab_size
        self.assertNotEqual(orig_vocab_size, 512)
        _model.resize_token_embeddings(512)
        self.assertEqual(_model.config.vocab_size, 512)
        model = SledModel(_model, SledConfig("facebook/bart-base", context_size=128))
        self.assertEqual(model.config.vocab_size, 512)
        self.assertEqual(_model.get_input_embeddings().weight.size()[0], 512)
        self.assertEqual(model.get_input_embeddings().weight.size()[0], 512)
        model.resize_token_embeddings(1024)
        self.assertEqual(model.config.vocab_size, 1024)
        self.assertEqual(_model.config.vocab_size, 1024)
        self.assertEqual(_model.get_input_embeddings().weight.size()[0], 1024)
        self.assertEqual(model.get_input_embeddings().weight.size()[0], 1024)

    def test_sled_model_parallel(self):
        assert torch.cuda.device_count() > 1
        model = SledModel(T5Model.from_pretrained("t5-small"), SledConfig("t5-small", context_size=512)).to("cuda:0")
        model.eval()
        assert model.is_parallelizable  # bart is not, only t5
        assert not model.model_parallel
        assert model.device_map is None

        model2 = SledModel(T5Model.from_pretrained("t5-small"), SledConfig("t5-small", context_size=512)).to("cuda:0")
        model2.eval()
        model2.parallelize()
        assert model2.model_parallel
        assert len(model2.device_map) == min(3, torch.cuda.device_count())

        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        input_ids = tokenizer([" ".join(list("1234567890"))] * 16, return_tensors="pt").input_ids.to("cuda:0")
        assert input_ids.size() == (16, 12)  # Batch size 16, input length 10 + BOS + EOS
        decoder_input_ids = tokenizer([" ".join(list("hello"))] * 16, return_tensors="pt").input_ids.to("cuda:0")
        assert decoder_input_ids.size() == (16, 11)  # Batch size 16, inputs length 5 + BOS + 4 spaces + EOS

        # simple verification there are no failures in the flow itself
        with torch.no_grad():
            outputs_expected = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone())
            outputs_got = model2(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone())
        self.compare_outputs_dict(
            outputs_expected, None, outputs_got, rtol=1e-5
        )  # the values may differ, but we need to make sure the sizes are correct

    def test_sled_with_data_parallel(self):
        assert torch.cuda.device_count() > 1
        model_ = SledModel(BartModel.from_pretrained("facebook/bart-base"),
                           SledConfig("facebook/bart-base", context_size=128)).to("cuda:0")
        model = nn.DataParallel(model_)
        model.eval()
        assert isinstance(model, nn.DataParallel)
        replicas = model.replicate(model.module, model.device_ids)
        assert len(replicas) == torch.cuda.device_count()
        for i, rep in enumerate(replicas):
            assert isinstance(rep, SledModel)
            assert rep.device.index == i
        forward = replicas[0].forward
        replicas[0].forward = "hello"
        assert replicas[0].forward == "hello"
        assert replicas[1].forward != "hello"
        replicas[0].forward = forward

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        input_ids = tokenizer([" ".join(list("1234567890"))] * 16, return_tensors="pt").input_ids.to("cuda:0")
        assert input_ids.size() == (16, 12)  # Batch size 16, input length 10 + BOS + EOS
        decoder_input_ids = tokenizer([" ".join(list("hello"))] * 16, return_tensors="pt").input_ids.to("cuda:0")
        assert decoder_input_ids.size() == (16, 7)  # Batch size 16, inputs length 5 + BOS + EOS

        # simple verification there are no failures in the flow itself
        with torch.no_grad():
            outputs_expected = model_(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone())
            outputs_got = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone())
        self.compare_outputs_dict(
            outputs_expected, None, outputs_got, rtol=1e-5
        )  # the values may differ, but we need to make sure the sizes are correct

    def test_sled_with_input_prefix(self):
        rtol = 1e-5
        model_, tokenizer = BartModel.from_pretrained("facebook/bart-base"), BartTokenizer.from_pretrained("facebook/bart-base")
        model = SledModel(model_, SledConfig("facebook/bart-base", context_size=16, prepend_prefix=True))
        model.eval()  # only change the model to be in eval (inference) mode, thus not changing layer_norm params and removing dropout

        document_input_ids = tokenizer(
            "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ).input_ids  # Batch size 1
        input_prefix_ids = tokenizer("What did studies show?\n\n", return_tensors="pt").input_ids  # Batch size 1
        decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        input_ids = torch.concat((input_prefix_ids, document_input_ids), dim=-1)
        prefix_length = torch.LongTensor([[input_prefix_ids.size(1)]])

        # simple verification there are no failures in the flow itself
        with torch.no_grad():
            _ = model(input_ids=input_ids.clone(), prefix_length=prefix_length.clone(),
                      decoder_input_ids=decoder_input_ids.clone())
            _ = model(input_ids=input_ids.clone(), prefix_length=prefix_length.clone(),
                      decoder_input_ids=decoder_input_ids.clone(),
                      return_dict=None)
            outputs_dict = model(input_ids=input_ids.clone(), prefix_length=prefix_length.clone(),
                                 decoder_input_ids=decoder_input_ids.clone(), return_dict=True)
            outputs_no_dict = model(input_ids=input_ids.clone(), prefix_length=prefix_length.clone(),
                                    decoder_input_ids=decoder_input_ids.clone(), return_dict=False)

        # let's verify that if the sequence is short, we behave exactly as the base model
        model = SledModel(model_, SledConfig("facebook/bart-base", context_size=512, prepend_prefix=False))
        # treat the while input as a single document and make sure the context size is large enough to contain it wholly
        model.eval()
        with torch.no_grad():
            output_expected = model_(
                input_ids=input_ids.clone(),
                decoder_input_ids=decoder_input_ids.clone(), return_dict=False
            )

        # on non dict return type
        with torch.no_grad():
            output_got = model(input_ids=input_ids.clone(), prefix_length=prefix_length.clone(),
                               decoder_input_ids=decoder_input_ids.clone(), return_dict=False)
        self.assertEqual(type(output_expected), type(output_got))
        self.assertEqual(type(output_expected), type(outputs_no_dict))
        self.assertEqual(len(output_got), len(output_expected))  # should be tuple so it's ok
        self.assertEqual(len(output_got), len(outputs_no_dict))  # should be tuple so it's ok
        _compare_tuple_of_tensors(self, output_expected, output_got, outputs_no_dict, rtol=rtol)

        # on dict return type
        with torch.no_grad():
            output_expected = model_(input_ids=input_ids.clone(),
                                     decoder_input_ids=decoder_input_ids.clone(), return_dict=True
            )
            output_got = model(input_ids=input_ids.clone(), decoder_input_ids=decoder_input_ids.clone(), return_dict=True)
        self.compare_outputs_dict(output_expected, output_got, outputs_dict, rtol=rtol)

    @unittest.expectedFailure
    def test_sled_with_variable_sized_input_prefix(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_sled_overhead(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_sled_multiple_update_steps(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_eval_mode(self):
        raise NotImplementedError
        # TODO - assert that eval() works by looking at the dropout rate? Three forward passes with the model in train,
        #  then three in eval and see that the train differ, but eval does not?

    @unittest.expectedFailure
    def test_prefix_prepending(self):
        # TODO - first, test the expected versions (no prepending+ no prefix given or w/ prepending+prefix given)
        # todo - test sled that should prepend but doesn't get it
        # todo - test sled that shouldn't prepend but gets a prefix length (make sure it ignores it)
        raise NotImplementedError

    @unittest.expectedFailure
    def test_drop_prefix_encoding(self):
        raise NotImplementedError



@require_torch
class SLEDForConditionalGenerationTest(unittest.TestCase):
    def test_facade_get_attr_behavior(self):
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        sled_model = SledForConditionalGeneration(base_model, SledConfig("t5-small", context_size=4))
        self.assertEqual(
            base_model.shared, sled_model.shared
        )  # sled model does not have a 'shared' attribute, only it's base model does

    def test_sled_for_cg_on_t5(self):
        self._run_sled_for_cg_model_test_case(
            T5ForConditionalGeneration.from_pretrained("t5-small"), T5Tokenizer.from_pretrained("t5-small"), "t5-small"
        )

    def test_sled_for_cg_on_bart(self):
        self._run_sled_for_cg_model_test_case(
            BartForConditionalGeneration.from_pretrained("facebook/bart-base"),
            BartTokenizer.from_pretrained("facebook/bart-base"), "facebook/bart-base"
        )

    def test_sled_for_cg_generate_on_t5(self):
        self._run_sled_for_cg_model_generate_test_case(
            T5ForConditionalGeneration.from_pretrained("t5-small"), T5Tokenizer.from_pretrained("t5-small"), "t5-small"
        )

    def test_sled_for_cg_generate_on_bart(self):
        self._run_sled_for_cg_model_generate_test_case(
            BartForConditionalGeneration.from_pretrained("facebook/bart-base"),
            BartTokenizer.from_pretrained("facebook/bart-base"), "facebook/bart-base"
        )

    def _run_sled_for_cg_model_test_case(self, model_, tokenizer, underlying_config: str, rtol=1e-5):
        model = SledForConditionalGeneration(model_, SledConfig(underlying_config, context_size=4))
        model.eval()

        input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>", return_tensors="pt").input_ids

        # simple verification there are no failures in the flow itself
        with torch.no_grad():
            _ = model(input_ids=input_ids.clone(), labels=labels.clone())
            _ = model(input_ids=input_ids.clone(), labels=labels.clone(), return_dict=None)
            outputs_dict = model(input_ids=input_ids.clone(), labels=labels.clone(), return_dict=True)
            outputs_no_dict = model(input_ids=input_ids.clone(), labels=labels.clone(), return_dict=False)

        # let's verify that if the sequence is short, we behave exactly as the base model
        model = SledForConditionalGeneration(model_, SledConfig(underlying_config, context_size=512))
        output_expected = model_(input_ids=input_ids.clone(), labels=labels.clone(), return_dict=False)

        # on non dict return type
        with torch.no_grad():
            output_got = model(input_ids=input_ids.clone(), labels=labels.clone(), return_dict=False)
        self.assertEqual(type(output_expected), type(output_got))
        self.assertEqual(type(output_expected), type(outputs_no_dict))
        self.assertEqual(len(output_got), len(output_expected))  # should be tuple so it's ok
        self.assertEqual(len(output_got), len(outputs_no_dict))  # should be tuple so it's ok
        _compare_tuple_of_tensors(self, output_expected, output_got, outputs_no_dict, rtol=rtol)

        # on dict return type
        with torch.no_grad():
            output_expected = model_(input_ids=input_ids.clone(), labels=labels.clone(), return_dict=True)
            output_got = model(input_ids=input_ids.clone(), labels=labels.clone(), return_dict=True)
        self.assertEqual(type(output_expected), type(output_got))
        self.assertEqual(type(output_expected), type(outputs_dict))
        self.assertListEqual(list(output_got.keys()), list(output_expected.keys()))
        self.assertListEqual(list(output_got.keys()), list(outputs_dict.keys()))
        for key in output_got.keys():
            if isinstance(output_got[key], torch.Tensor):
                self.assertTrue(torch.allclose(output_got[key], output_expected[key], rtol=rtol))
                # we can't expect the values to be the same when different context length, but at least can verify shapes
                self.assertTrue(output_got[key].size() == outputs_dict[key].size())
            elif isinstance(output_got[key], tuple):
                _compare_tuple_of_tensors(self, output_got[key], output_expected[key], outputs_dict[key], rtol=rtol)

    def _run_sled_for_cg_model_generate_test_case(self, model_, tokenizer, underlying_config: str, rtol=1e-5):
        input_ids = tokenizer(
            "summarize: studies have shown that owning a dog is good for you ", return_tensors="pt"
        ).input_ids  # Batch size 1
        model_.eval()
        # once it is used in SledForConditionalGeneration, it cannot generate directly anymore
        output_expected = model_.generate(input_ids.clone())

        model = SledForConditionalGeneration(model_, SledConfig(underlying_config, context_size=4))
        model.eval()

        # simple verification there are no failures in the flow itself
        _ = model.generate(
            torch.cat((input_ids.clone(), input_ids.clone()))
        )  # just to make sure can generate over two sequences
        _ = model.generate(input_ids.clone())
        outputs_no_dict = model.generate(input_ids.clone())
        assert outputs_no_dict.dim() == 2
        assert outputs_no_dict.size()[0] == 1

        # let's verify that if the sequence is short, we behave exactly as the base model
        model = SledForConditionalGeneration(model_, SledConfig(underlying_config, context_size=512))
        model.eval()

        output_got = model.generate(input_ids.clone())
        self.assertEqual(type(output_expected), type(output_got))
        self.assertEqual(type(output_expected), type(outputs_no_dict))
        self.assertEqual(len(output_got), len(output_expected))  # should be tuple so it's ok
        self.assertEqual(len(output_got), len(outputs_no_dict))  # should be tuple so it's ok
        # no point checking the dim as the generation length may be different
        _compare_tuple_of_tensors(self, (output_expected,), (output_got,), None, rtol=rtol)
