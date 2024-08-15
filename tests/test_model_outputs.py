# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/8/15
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_openmoe import HFOpenMoeForCausalLM


class ModelOutputTest(unittest.TestCase):
    def testModelOutputLogits(self):
        official_model = AutoModelForCausalLM.from_pretrained(
            "OrionZheng/openmoe-8b", device_map="auto"
        )
        hf_model = HFOpenMoeForCausalLM.from_pretrained(
            "Phando/openmoe-8b-native-pt", device_map="auto"
        )
        inputs = {
            "input_ids": torch.randint(low=10000, high=20000, size=(4, 16)).cuda(),
            "attention_mask": torch.ones(4, 16).cuda()
        }
        official_outputs = official_model(**inputs)
        hf_outputs = hf_model(**inputs)

        self.assertTrue(
            torch.allclose(official_outputs.logits, hf_outputs.logits)
        )

    def testTokenizer(self):
        official_tokenizer = AutoTokenizer.from_pretrained("OrionZheng/openmoe-8b")
        hf_tokenizer = AutoTokenizer.from_pretrained("Phando/openmoe-8b-native-pt")
        text = "Shall I compare thee to a summer's day?"

        self.assertTrue(torch.allclose(
            official_tokenizer(text)["input_ids"],
            hf_tokenizer(textt)["input_ids"]
        ))
