# ReaLM: Reliable and Efficient Large Language Model Inference with Statistical Algorithm-Based Fault Tolerance
## Abstract
The demand for efficient large language model (LLM) inference has propelled the development of dedicated accelerators. As accelerators are vulnerable to hardware faults due to aging, variation, etc, existing accelerator designs often reserve a large voltage margin or leverage algorithm-based fault tolerance (ABFT) techniques to ensure LLM inference correctness. However, previous methods often overlook the inherent fault tolerance of LLMs, leading to high computation and energy overhead. To enable reliable yet efficient LLM inference, in this paper, we propose a novel algorithm/circuit co-design framework, dubbed ReaLM. For the first time, we systematically characterize the fault tolerance of LLMs by performing a large scale error injection study of representative LLMs and natural language understanding tasks. Then, we propose a statistical ABFT algorithm that fully leverages the error robustness to minimize error recovery as much as possible. We also customize the error detection circuits to enable a low-cost online collection of error statistics. Extensive experiments show that with only 1.42% circuit area and 1.79% power overhead, our ReaLM can reduce perplexity degradation from 18.54 to 0.29. Compared to existing methods, ReaLM consistently reduces recovery costs across different operating voltages and improves energy efficiency by up to 35.83% without compromising LLM performance.
## clarification
In this project, we used the method of **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"** (ICML 2023) to quantize LLM. Therefore, when running this project, you need to first pull the smoothquant project from <https://github.com/mit-han-lab/smoothquant>. Of course, we have integrated the smoothquant project into this project, you can just download our project.
## Initialization
**Same as smoothquant's Installation**
```python
conda create -n smoothquant python=3.8
conda activate smoothquant
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.36.0 accelerate datasets zstandard

python setup.py install
```
## Usage
Running our project is very simple. The error_model folder contains the model that injects errors. You can directly run any Python code in this folder, for example
```python
python error_model/smoothquant_Llama2_error_inject.py(any python code you want to run)
```
## Illustratation
We also studied the fault tolerance characteristics of the LLM decode stage. In order to make the model perform better when no errors are injected, we performed prompt operations on models during testing. For details, see the contents of the prompt directory.
