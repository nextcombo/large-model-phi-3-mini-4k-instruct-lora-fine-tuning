# [Large Model] Phi-3-mini-4k-Instruct Lora fine-tuning

This section briefly introduces how to base on transformers, peft and other frameworks, Phi-3-mini-4k-Inst.

In this section, we will briefly introduce how to base on transformers. Frameworks such as PEFT. Perform LoRa micro-tuning on the Phi-3-mini-4k-Instruct model. Lora is an efficient fine-tuning method. For a deeper understanding of its principles, refer to the blog: Lora | Deep Understanding.

This tutorial will provide a nodebook file in the same directory for everyone to better learn from.

### Environmental Preparation

Rent a 3090-class 24G VRAM graphics card machine on the Autodl platform, as shown in the image, select the mirror by `PyTorch-->2.1.0-->3.10(ubuntu22.04)-->12.1`.
Next, open JupyterLab on the server you just rented and start configuring the environment, downloading models, and running demos.

![](01-6dffb529.png)

### Environment Configuration

After completing the basic environment configuration and local model deployment, you also need to install some third-party libraries. You can use the following command:

-
-
-
-
-
-
-
-
-
-
-
-
-
-

```
python -m pip install --upgrade pip# Change pypi Installation of the Source Acceleration Librarypip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope==1.9.5pip install "transformers>=4.40.0"pip install streamlit==1.24.0pip install sentencepiece==0.1.99pip install accelerate==0.29.3pip install datasets==2.19.0pip install peft==0.10.0
MAX_JOBS=8 pip install flash-attn --no-build-isolation

```

>

Considering some students may encounter issues with their environment setup, we have prepared a Phi-3 environment image for the AutoDL platform, which is suitable for all deployment environments in this repository. Click the link below and create an AutoDL example directly.
*https://www.codewithgpu.com/i/datawhalechina/self-llm/Phi-3-Lora*

In this section, we will fine-tune the dataset to be placed in the root directory /dataset.

### Model Download

Download the model using the snapshot_download function in ModelScope. The first parameter is the model name, and the second parameter is the cache directory for the model download.

Create a new file named `model_download.py` under the /root/autodl-tmp directory and enter the following content:

-
-
-
-
-
-

```
import torchfrom modelscope import snapshot_download, AutoModel, AutoTokenizerimport os
model_dir = snapshot_download('LLM-Research/Phi-3-mini-4k-instruct', cache_dir='/root/autodl-tmp', revision='master')

```

### Instruction Set Construction

The fine-tuning of LLM generally refers to the instructional fine-tuning process. As mentioned, the micro-tuning data we use is in the form:

-
-
-
-
-
-

```
{ "instruction": "Answer the following user question, outputting only the answer.", "input": "1+1 equals to what?", "output": "2"}

```

Among them, `instruction` is the user's instruction, which informs the model of the task it needs to complete; `input` is the user's input, which is the necessary content for completing the user's instruction; `output` is the model's output.

Our core training objective is to enable the model to understand and follow user instructions. Therefore, when constructing an instruction set, we should tailor it to our target task. For instance, in this section, we use the Chat-Zhenhuan project co-authored by the author as an example. Our goal is to build a personalized LLM that can simulate the dialogue style of Zhenhuan. Consequently, the instructions we construct are similar to:

-
-
-
-
-
-

```
{ "instruction": "Who are you?", "input": "", "output": "My father is Zhen Yuandao, the Deputy Prefect of the Dali Temple.

```

All the instruction datasets we constructed are located in the root directory.

### Data Formatting

The training data for `Lora` requires formatting and encoding before being input into the model for training. Those familiar with the Pytorch model training process will know that we generally need to encode the input and output texts of an input sample into `input_ids` and `labels`, respectively, and the resulting encoded results are multi-dimensional vectors. We first define a preprocessing function that is used to encode each sample's input and output texts and return an encoded dictionary:

-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-

```
def process_func(example):    MAX_LENGTH = 384    # LlamaThe tokenizer will split a Chinese character into multiple parts.token，Therefore, some maximum lengths need to be relaxed to ensure the integrity of data.    input_ids, attention_mask, labels = [], [], []    instruction = tokenizer(f"<|user|>\n{example['instruction'] + example['input']}<|end|>\n<|assistant|>\n", add_special_tokens=False)  # add_special_tokens Do not add anything. special_tokens    response = tokenizer(f"{example['output']}<|end|>\n", add_special_tokens=False)    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # Becauseeos tokenWe also need to pay attention to it, so...1    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]    if len(input_ids) > MAX_LENGTH:  # Make a cut        input_ids = input_ids[:MAX_LENGTH]        attention_mask = attention_mask[:MAX_LENGTH]        labels = labels[:MAX_LENGTH]    return {        "input_ids": input_ids,        "attention_mask": attention_mask,        "labels": labels    }

```

The `Prompt Template` used in `Phi-3-mini-4k-Instruct` is as follows:

-
-
-
-
-
-
-

```
<|system|>You are a helpful assistant<|end|><|user|>Who are you?<|end|><|assistant|>I am a useful assistant.<|end|>

```

### Loading tokenizer and half-precision model

The model is loaded in half-precision format. If your GPU is relatively new, you can load it using `torch.bfolat`. For custom models, make sure to specify the `trust_remote_code` parameter as `True`.

-
-
-
-

```
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct', use_fast=False, trust_remote_code=True)tokenizer.pad_token = tokenizer.eos_tokemodel = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct', device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)

```

### Define LoraConfig

The `LoraConfig` class can set many parameters, but there are not many main parameters. Let me briefly explain them for those interested in the source code.

-

`task_type`: Model type

-

`target_modules`: The name of the model layer that needs to be trained, mainly the layers in the `attention` part. Different models correspond to different names for these layers. It can be passed as an array, a string, or a regular expression.

-

`r`: The rank of `lora`, which can be seen in the principle of `Lora`.

-

`lora_alpha`: `Lora alaph`, see the principle of `Lora` for specific functions.

What is the scaling of `Lora`? Of course, it's not `r`, which is the rank. The scaling here is `lora_alpha/r`, and in this `LoraConfig`, the scaling is 4 times.

-
-
-
-
-
-
-
-
-

```
config = LoraConfig(   task_type=TaskType.CAUSAL_LM,   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  inference_mode=False,  r=8,  lora_alpha=32,  # Lora alaph, specific role see Lora principle  lora_dropout=0.1# Dropout ratio)

```

### Customize TrainingArguments Parameters

The source code of the `TrainingArguments` class also explains the specific roles of each parameter. Of course, everyone can explore on their own, and here I will simply mention a few commonly used ones.

-

`output_dir`: The output path of the model

-

`per_device_train_batch_size`: As the name suggests, it is equivalent to `batch_size`.

-

`gradient_accumulation_steps`: Gradient accumulation. If your GPU memory is limited, you can set `batch_size` smaller and increase the gradient accumulation step accordingly.

-

`logging_steps`: how many steps, output once `log`

-

`num_train_epochs`: As the name suggests, `epoch`

-

`gradient_checkpointing`: Gradient Checkpointing. Once enabled, the model must execute `model.enable_input_require_grads()`, which can be explored by everyone on their own. Here, I will not elaborate further.

-
-
-
-
-
-
-
-
-
-
-
-

```
args = TrainingArguments(    output_dir="./output/Phi-3",    per_device_train_batch_size=4,    gradient_accumulation_steps=4,    logging_steps=10,    num_train_epochs=3,    save_steps=100,    learning_rate=1e-4,    save_on_each_node=True,    gradient_checkpointing=True)

```

### Using Trainer for Training

-
-
-
-
-
-
-
-
-

```
trainer = Trainer(    model=model,    args=args,    train_dataset=tokenized_id,    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),)trainer.train()

```

### Save lora weights

-
-
-
-

```
lora_path='./Phi-3_lora'trainer.model.save_pretrained(lora_path)tokenizer.save_pretrained(lora_path)

```

### Loading LoRa Weighted Inference

After training, you can load the `lora` weights for inference using the following method:

-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-

```
from transformers import AutoModelForCausalLM, AutoTokenizerimport torchfrom peft import PeftModel
model_path = '/root/autodl-tmp/LLM-Research/Phi-3-mini-4k-instruct'lora_path = './Phi-3_lora' # loraWeighted path
# Loadingtokenizertokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
# Load modelmodel = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
# Load Lora weight model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)
prompt = "Who are you?"messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
generated_ids = model.generate(    model_inputs.input_ids,    max_new_tokens=512,    eos_token_id=tokenizer.encode('<|endoftext|>')[0])
outputs = generated_ids.tolist()[0][len(model_inputs[0]):]response = tokenizer.decode(outputs).split('<|end|>')[0]
print(response)

```

Completely fine-tuned code download link:
https://github.com/datawhalechina/self-llm/blob/master/phi-3/Phi-3-mini-4k-Instruct-Lora.ipynb


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
