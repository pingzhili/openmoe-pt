# openmoe-pt
An unofficial OpenMoE reproduction with native PyTorch in the HuggingFace style.

The official PyTorch implementation, with Colossal AI, can be found in [OpenMoE](https://github.com/XueFuzhao/OpenMoE).

My reproduction focuses on simplifying integration with HuggingFace and other codebases.


## Example

```python
from hf_openmoe import HFOpenMoeForCausalLM, HFOpenMoeTokenizer

model = HFOpenMoeForCausalLM.from_pretrained("Phando/openmoe-8b-native-pt")
tokenizer = HFOpenMoeTokenizer.from_pretrained("Phando/openmoe-8b-native-pt")
```

## Citation

Please cite the official [OpenMoE](https://github.com/XueFuzhao/OpenMoE) if you use the model and code in this repo.

```bibtex
@article{xue2024openmoe,
  title={OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models},
  author={Xue, Fuzhao and Zheng, Zian and Fu, Yao and Ni, Jinjie and Zheng, Zangwei and Zhou, Wangchunshu and You, Yang},
  journal={arXiv preprint arXiv:2402.01739},
  year={2024}
}
```