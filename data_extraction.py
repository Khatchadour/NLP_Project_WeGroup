"""Extract data from documents using pre-trained models."""

#import deepspeed
import json
import pytesseract
import re
import time
import torch
import yaml

from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


_TESSERACT_CONFIG = r'--oem 3 --psm 6'


# Timing decorator
def time_func(func):
    def wrapper(*args, **kwargs):
        t_1 = time.time()
        res = func(*args, **kwargs)
        t_2 = time.time()
        t = f"Function: {func.__name__} took {(t_2-t_1):.2f}s to run."
        return t, *res
    return wrapper


def load_yaml():
    """Loads question and document path from YAML file for examples."""
    with open('prompts.yml', 'r') as file:
        yaml_content = file.read()
    data = yaml.safe_load(yaml_content)
    return data


@time_func
def preload_models(model_name, model_type):
    """Loads models and tokenizers.

    This method loads the necessary models and tokenizers for inference according to the selected model.

    Arguments
        model_name: model path as specified on HuggingFace hub.
        model_type: type of model to load. Either question answering ('qa') 
                    or causal language modelling ('lm') framework.

    Returns
    -------
        A tuple with tokenizer and model object.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name) #eos_token="."
    kwargs = {       
        'pretrained_model_name_or_path': model_name,
        'device_map': "auto",
        #'load_in_8bit': True,
        #'max_memory': {0: "24GIB"},
        #'torch_dtype': torch.float16,
        #'llm_int8_threshold': 0,
        #'ds_model': deepspeed.init_inference,
    }
    if model_type == 'qa':
        model = AutoModelForSeq2SeqLM.from_pretrained(**kwargs)
    elif model_type == 'lm':
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
    else:
        raise ValueError("Invalid model! Use one of the BLOOM family of models.")
    return tokenizer, model


@time_func
def extract_text(filename):
    """Uses OCR on pdf document.

    This method loads pdf document, does OCR on it and provides an output as a raw string. 
    Can handle multiple pages (but requires care). In principle can be any other OCR engine aside from tesseract. 

    Arguments
        filename: local document path.

    Returns
    -------
        A list with one element: string of extracted data.
    """
    pages = convert_from_path(filename)
    ocr_output = ''
    for doc in pages:
        ocr_output += pytesseract.image_to_string(doc, config=_TESSERACT_CONFIG) + '\n'
    ocr_output = [ocr_output]
    return ocr_output


def generate_prompt(text, question):
    """Generates prompt for model."""
    prompt = ['Insurance policy document: ' + '\n' + '<(<' + text + '>)>' + '\n' + question + '\n']      
    #prompt = ['Insurance policy document: ' + '\n' + text + '\n' + question + '\n']      
    return prompt


@time_func
def get_completion(prompt, tokenizer, model):
    """Generates output.

    This method tokenizes the prompt, uses it in the model to generate an output and 
    decode it in a human readable textual format.

    Arguments
        prompt: prompt to use as input.
        tokenizer: tokenizer object.
        model: model object.

    Returns
    -------
        Textual output as a string.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        #early_stopping=True,
        min_length=1,
        #length_penalty=0.1,
        #eos_token_id=tokenizer.eos_token_id
        #temperature=0, 
        #max_length=1024,
    ) 
    response = [tokenizer.decode(output[0])]
    return response


def parse_response(response):
    """Parses response to JSON format."""
    start_parse = response.find('{')
    end_parse = response.rfind('}')
    response_upd = [response[start_parse:(end_parse+1)]]
    if start_parse == -1 or end_parse == -1:
        raise ValueError("Output was not generated in a suitable format for JSON file.") 
    return response_upd


def extract_data_from_file_example(model_name, model_type):
    """"A sequence of steps to extract data from a document to test."""
    t1, tokenizer, model = preload_models(model_name, model_type)
    data = load_yaml()
    q = data[0]['question']
    doc = data[1]['doc']
    t2, text = extract_text(doc + ".pdf")
    prompt = generate_prompt(text, q)
    t3, response = get_completion(prompt[0], tokenizer, model)
    print(t1, t2, t3)
    print(response)


def extract_data_from_file(model_name, model_type, document_name, path_to_save):
    """"A sequence of steps to extract data from a specified document to save as a JSON file."""
    _, tokenizer, model = preload_models(model_name, model_type)
    data = load_yaml()
    q = data[0]['question']
    _, text = extract_text(document_name + ".pdf")
    prompt = generate_prompt(text, q)
    _, response = get_completion(prompt[0], tokenizer, model)
    response = parse_response(response)
    response_dict = json.loads(response[0])
    with open(path_to_save, 'w') as json_file:
        json.dump(response_dict, json_file)
