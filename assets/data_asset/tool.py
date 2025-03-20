import torch
import transformers
# from torch.utils.data import Dataset
from datasets import Dataset
from transformers import Trainer,TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
# Load model directly
import torch.nn as nn
from huggingface_hub import upload_file

from IPython.display import display, Markdown
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import numpy as np
import os
import json

from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"]="0"
loading=load_dotenv('.env')

PATH = str(os.getenv('PATH_DATASET'))
DEVICE = 'cuda' 
# print(PATH)if torch.cuda.is_available() else 'cpu'

# DEFAULT_PAD_TOKEN = "[PAD]"



def Load_json(file_name=None, file_path=PATH,size=0.5,validation=False):
    file_path = os.path.join(file_path, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = data[:int(len(data)*size)]
    print("Data loaded successfully and it contains:", len(data))
    print(data[0].keys())
    if validation:
        data = data[:int(len(data)*size)]
        validation_data = data[int(len(data)*size):]
        return data, validation_data
    return data


def create_prompt(question, context=None):
    PROMPT_DICT = {
        "prompt_context": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:"
        ),
        "prompt_no_context": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}\n\n### Response:"
        ),
    }
    return PROMPT_DICT["prompt_context"] if context else PROMPT_DICT["prompt_no_context"]

def create_prompt_with_answer(question, context, answer):
    PROMPT_DICT = {
        "prompt_context": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}\n\n### Input:\n{context}\n\n### Response:\n{answer}"
        ),
        "prompt_no_context": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{question}\n\n### Response:\n{answer}"
        ),
    }
    return PROMPT_DICT["prompt_context"] if context else PROMPT_DICT["prompt_no_context"]

def create_prompt_with_answer_v2(question, answer, context=None, EOS_TOKEN='</s>'):
    PROMPT_DICT = {
        "prompt_context": (
            f"### QUESTION\n{question}\n\n### CONTEXT\n{context}\n\n### ANSWER\n{answer}{EOS_TOKEN}"

        ),
        "prompt_no_context": (
            f"### QUESTION\n{question}\n\n### ANSWER\n{answer}{EOS_TOKEN}"
        ),
    }
    return PROMPT_DICT["prompt_context"] if (context != None or context!='')  else PROMPT_DICT["prompt_no_context"]

def create_prompt_with_answer_v3(question, answer, context=None, EOS_TOKEN='</s>'):
    PROMPT_DICT = {
        "prompt_context": (
            "Below is an instruction (question) that describes a task, paired with an input (context) that provides further context. "
            "Write an SQL query response that appropriately completes the request.\n\n"
            f"### QUESTION\n{question}\n\n### CONTEXT\n{context}\n\n### ANSWER\n{answer}{EOS_TOKEN}"

        ),
        "prompt_no_context": (
            "Below is an instruction (question) that describes a task. "
            "Write an SQL query response that appropriately completes the request.\n\n"
            f"### QUESTION\n{question}\n\n### ANSWER\n{answer}{EOS_TOKEN}"
        ),
    }
    return PROMPT_DICT["prompt_context"] if (context != None or context!='')  else PROMPT_DICT["prompt_no_context"]


def create_prompt_v2(question, context=None, EOS_TOKEN='</s>'):
    PROMPT_DICT = {
        "prompt_context": (
            f"### QUESTION\n{question}\n\n### CONTEXT\n{context}\n\n### ANSWER\n"
        ),
        "prompt_no_context": (
            f"### QUESTION\n{question}\n\n### ANSWER\n"
        ),
    }
    return PROMPT_DICT["prompt_context"] if (context != None or context!='') else PROMPT_DICT["prompt_no_context"]

def create_prompt_v3(question, context=None, EOS_TOKEN='</s>'):
    PROMPT_DICT = {
        "prompt_context": (
            "Below is an instruction (question) that describes a task, paired with an input (context) that provides further context. "
            "Write an SQL query response that appropriately completes the request.\n\n"
            f"### QUESTION\n{question}\n\n### CONTEXT\n{context}\n\n### ANSWER\n"
        ),
        "prompt_no_context": (
            "Below is an instruction (question) that describes a task. "
            "Write an SQL query response that appropriately completes the request.\n\n"
            f"### QUESTION\n{question}\n\n### ANSWER\n"
        ),
    }
    return PROMPT_DICT["prompt_context"] if (context != None or context!='') else PROMPT_DICT["prompt_no_context"]

# test create_prompt
# print(create_prompt("Who is Napoleon Bonaparte?"))
# print('------------'*10)
# print(create_prompt("Who is Napoleon Bonaparte?", "Napoleon Bonaparte was a French military and political leader."))

class Model_params():
    def __init__(self, model_name=None, device_map="auto", use_cache=False, pretraining_tp=0, quatization=None, transformer_from="auto", **kwargs):
        self.model_name = model_name
        self.device_map = device_map
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.revision = kwargs.get('revision', "main")
        # load_in_16bit=torch.float16
        self.load_in_16bit = kwargs.get('load_in_16bit',None)
        self.load_in_8bit = kwargs.get('load_in_8bit',False)
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.token = kwargs.get('token', None)
        self.quatization = quatization
        self.model_from = {'model':AutoModelForCausalLM, 'tokenizer':AutoTokenizer}
        if quatization == '4bit':
            #4bit quantization:
            self.quatization = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                # bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype = torch.float16,
            )
        elif quatization == '8bit':
                #8bit quantization:
            self.quatization = BitsAndBytesConfig(
                load_in_8bit=True,  # Enable 8-bit quantization
                # llm_int8_threshold=6.0,  # This threshold helps handle outlier values
                bnb_8bit_compute_dtype=torch.float16,  # Ensure correct dtype
                bnb_8bit_use_double_quant = True,
                bnb_8bit_quant_type = "int8",
            )
        else:
            self.quatization =  None
        self.transformer_from = transformer_from
        if transformer_from == "llama":
            self.model_from = {'model':LlamaForCausalLM, 'tokenizer':LlamaTokenizer}
        else:
            self.model_from = {'model':AutoModelForCausalLM, 'tokenizer':AutoTokenizer}

    def load_model(self):
        if self.quatization != None:
            model = self.model_from['model'].from_pretrained(self.model_name, 
                                                            device_map=self.device_map,
                                                            revision=self.revision,
                                                            trust_remote_code=self.trust_remote_code,
                                                            quantization_config = self.quatization,
                                                            token = self.token,
                                                        )
        elif self.load_in_16bit != None:
            model = self.model_from['model'].from_pretrained(self.model_name, 
                                                            device_map=self.device_map,
                                                            revision=self.revision,
                                                            trust_remote_code=self.trust_remote_code,
                                                            torch_dtype = self.load_in_16bit,
                                                            token = self.token,
                                                        )
        else:
            model = self.model_from['model'].from_pretrained(self.model_name, 
                                                            device_map=self.device_map,
                                                            revision=self.revision,
                                                            trust_remote_code=self.trust_remote_code,
                                                            token = self.token,
                                                        )
        model.config.use_cache = self.use_cache
        model.config.pretraining_tp = self.pretraining_tp
        print("Model loaded successfully")
        return model

    def load_tokenizer(self):
        tokenizer = self.model_from['tokenizer'].from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # pad sequences
        tokenizer.padding_side = 'right'
        print("Tokenizer loaded successfully")
        return tokenizer

    def load_model_tokenizer(self):
        model = self.load_model()
        tokenizer = self.load_tokenizer()
        return model, tokenizer
    
def generate_text(model, tokenizer, input_text="def generate():", max_length=200, **kwargs):
    # 
    model.eval()
    # model.to(DEVICE)
    # change input text as desired
    # input_text = "def generate():"
    # tokenize the text
    input_tokens = tokenizer(input_text, return_tensors="pt")
    # transfer tokenized inputs to the device
    for i in input_tokens:
        input_tokens[i] = input_tokens[i].to(DEVICE)

    with torch.no_grad():
        # generate output tokens
        output = model.generate(**input_tokens, max_length=max_length, **kwargs)

    # with torch.cuda.amp.autocast():
    # generate output tokens
        # output = model.generate(**input_tokens, max_length=max_length, **kwargs)
    # decode output tokens into text
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    # loop over the batch to print, in this example the batch size is 1
    # for i in output:
    #     print(i)
    return output


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def pushing_model_to_HF(model_name, model, HF_user="OussamaAzz", **kwargs):
    model.push_to_hub(f'{HF_user}/{model_name}', **kwargs)


def load_model_from_HF(model_name, HF_user="OussamaAzz", quantization=True, **kwargs):
    peft_model_id = f'{HF_user}/{model_name}'
    config = ''
    config2 = PeftConfig.from_pretrained(peft_model_id)
    base_model_name =kwargs.get('base_model_name', config2.base_model_name_or_path)
    if quantization == '4bit':
        config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='auto', quantization_config=config, return_dict=True, torch_dtype = torch.float16)
    elif quantization == '8bit':
        config = BitsAndBytesConfig(
            load_in_8bit=True,  # Enable 8-bit quantization
            # llm_int8_threshold=6.0,  # This threshold helps handle outlier values
            bnb_8bit_compute_dtype=torch.float16,  # Ensure correct dtype
            bnb_8bit_use_double_quant = True,
            bnb_8bit_quant_type = "int8",
        )    
        model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='auto', quantization_config=config, return_dict=True, torch_dtype = torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='auto', return_dict=True, torch_dtype = torch.float16)
        

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # model = AutoModelForCausalLM.from_pretrained(config2.base_model_name_or_path, return_dict=True, device_map='auto')
    # model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl", quantization_config=config)
    sql_model = PeftModel.from_pretrained(model, peft_model_id)
    return model, tokenizer, sql_model


def make_inferece(model, tokenizer, data, sql_model, max_length=200, **kwargs):
    model.eval()
    batch = tokenizer(data, return_tensors='pt')
    sql_model = sql_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sql_model = sql_model.to(device)  # Now sql_model is recognized within the function scope
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Your inference code here
    with torch.cuda.amp.autocast():
        output_tokens = sql_model.generate(**batch, 
                                            max_new_tokens=max_length,
                                            # repetition_penalty=1.5,
                                            # temperature=0.9,
                                            **kwargs,
                                        )

    display(Markdown((tokenizer.decode(output_tokens[0], skip_special_tokens=True))))
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)


def generate_text_v2(model, tokenizer, input_text="def generate():", max_length=200, **kwargs):
    # Set the model to evaluation mode and move it to the desired device
    model.eval()
    model.to(DEVICE)

    # Tokenize the input text and move the tokens to the device
    input_tokens = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    # Use mixed precision for faster inference
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Generate output tokens
            output = model.generate(**input_tokens, 
                                    max_length=max_length, 
                                    
                                    **kwargs)

    # Decode the output tokens into text
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

    return output_text


import matplotlib.pyplot as plt
import numpy as np
from transformers import TrainerCallback

class PlottingCallback(TrainerCallback):
    def __init__(self, model_name):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_steps = []
        self.val_steps = []
        self.model_name = model_name

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(logs['loss'])
        if 'eval_loss' in logs:
            self.val_steps.append(state.global_step)
            self.val_losses.append(logs['eval_loss'])
        if 'eval_accuracy' in logs:
            self.val_accuracies.append(logs['eval_accuracy'])

    def plot_final_metrics(self, base_dir="plots", repo_id='koukoudzz/gpt2_sql-v0.1', path_in_repo='plots'):
        # Create the local directory based on the model name
        save_dir = os.path.join(base_dir, self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Define plot file names
        loss_plot_path = os.path.join(save_dir, 'loss_plot.png')
        overfitting_plot_path = os.path.join(save_dir, 'overfitting_plot.png')
        accuracy_plot_path = os.path.join(save_dir, 'accuracy_plot.png')
        perplexity_plot_path = os.path.join(save_dir, 'perplexity_plot.png')

        # Plot and save losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_steps, self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_steps, self.val_losses, label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(loss_plot_path)
        plt.show()
        plt.close()

        # Plot and save overfitting measure
        if self.val_losses:
            min_length = min(len(self.train_losses), len(self.val_losses))
            overfitting = np.array(self.train_losses[:min_length]) - np.array(self.val_losses[:min_length])
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_steps[:min_length], overfitting, color='red', label='Overfitting (Train Loss - Val Loss)')
            plt.xlabel('Step')
            plt.ylabel('Overfitting')
            plt.legend()
            plt.title('Overfitting Measure')
            plt.savefig(overfitting_plot_path)
            plt.show()
            plt.close()

        # Plot and save perplexity
        if self.train_losses or self.val_losses:
            plt.figure(figsize=(10, 6))
            train_perplexity = [torch.exp(torch.tensor(loss)).item() for loss in self.train_losses]
            plt.plot(self.train_steps, train_perplexity, label='Training Perplexity')
            if self.val_losses:
                val_perplexity = [torch.exp(torch.tensor(loss)).item() for loss in self.val_losses]
                plt.plot(self.val_steps, val_perplexity, label='Validation Perplexity')
            plt.xlabel('Step')
            plt.ylabel('Perplexity')
            plt.legend()
            plt.title('Training and Validation Perplexity')
            plt.savefig(perplexity_plot_path)
            plt.show()
            plt.close()

        # Upload files to Hugging Face Hub
        if repo_id and path_in_repo:
            try:
                upload_file(path_or_fileobj=loss_plot_path, path_in_repo=os.path.join(path_in_repo, 'loss_plot.png'), repo_id=repo_id)
                upload_file(path_or_fileobj=overfitting_plot_path, path_in_repo=os.path.join(path_in_repo, 'overfitting_plot.png'), repo_id=repo_id)
                upload_file(path_or_fileobj=perplexity_plot_path, path_in_repo=os.path.join(path_in_repo, 'perplexity_plot.png'), repo_id=repo_id)
                print(f"Plots successfully uploaded to {repo_id} under {path_in_repo}.")
            except Exception as e:
                print(f"Failed to upload plots to Hugging Face Hub: {e}")


# Function to convert data to Dataset
def convert_to_dataset(data, tokenizer, include_labels=True):
    # Assume `tokenizer` is already defined and imported
    EOS_TOKEN = tokenizer.eos_token  
    # Assuming `tool.create_prompt_with_answer_v2(**d)` returns a string
    text = [create_prompt_with_answer_v2(**d)  for d in data]
    question = [d['question'] for d in data]
    context = [d['context'] for d in data]
    answer = [d['answer'] for d in data]
    dict_data = {"question": question, "context": context, "answer": answer}
    dict_data = Dataset.from_dict(dict_data)
    
    if include_labels:
        # Creating new labels for the dataset
        labels = [i for i in range(len(data))]
        return Dataset.from_dict({"text": text, "labels": labels})
    else:
        return Dataset.from_dict({"text": text, "source": dict_data})
