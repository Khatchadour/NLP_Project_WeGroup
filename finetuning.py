"""Finetune models."""

import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling


_BLOCK_SIZE = 128


def preload(model_id, dataset_path):
    """Preload model, tokenizer, dataset and define model type."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {
        'pretrained_model_name_or_path': model_id,
        'device_map': "auto"
    }
    if model_id.startswith("bigscience/mt"):
        model = AutoModelForSeq2SeqLM.from_pretrained(**kwargs)
        model_type = 'qa'
    elif model_id.startswith("bigscience/bloom"):
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        model_type = 'lm'
    else:
        raise ValueError("Invalid model! Use one of the BLOOM family of models.")
    dataset = load_dataset(dataset_path)
    return dataset, tokenizer, model, model_type


def max_length_est(dataset, tokenizer, model_type):
    """Estimates max_length.

    This method estimates max_length to use in tokenizing input for truncation and padding.
    We determine the longest and average input length in the dataset and 
    use them to calculate some threshold to use as max_length.

    Arguments
        dataset: Dataset object.
        tokenizer: tokenizer object.
        model_type: type of model to load. Either question answering ('qa') 
                    or causal language modelling ('lm') framework.

    Returns
    -------
        If model_type is 'qa', returns integer max_length according to calculations. Else, returns None.
    """
    if model_type == 'qa':
        examples = []
        for example in dataset['train']['question']:  
            tokenized_example = tokenizer(example)
            examples.append(len(tokenized_example['input_ids']))
        avg_length = np.mean(examples)
        max_length = int((max(examples) - avg_length)/2 + avg_length)
        return max_length
    else:
        return None


def preprocess(dataset, tokenizer, model_type, batch_size=16, num_proc=8, max_length=1000):
    """Tokenize dataset.

    This method does tokenizing of the dataset by applying corresponding functions depending on model type.

    Arguments
        dataset: Dataset object.
        tokenizer: tokenizer object.
        model_type: type of model to load. Either question answering ('qa') 
                    or causal language modelling ('lm') framework.
        batch_size: batch size for map processing of datasets.
        num_proc: number of processes for map processing of datasets.
        max_length: max_length to use in tokenizing input for truncation and padding.

    Returns
    -------
        Returns tokenized dataset object.
    """
    if model_type == 'qa':
        tokenized_dataset = dataset.map(
            tokenize_data,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=['question', 'answer'],
            fn_kwargs={'tokenizer': tokenizer, 'model_type': model_type, 'max_length': max_length}
        )
    elif model_type == 'lm':
        #dataset.reset_format()
        dataset = dataset.map(
            lambda x: {'text': x['question']+x['answer']},
            # batched=True,
            # batch_size=batch_size,
            # num_proc=num_proc,
            remove_columns=['question', 'answer']
        )
        tokenized_dataset = dataset.map(
            tokenize_data,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=['text'], 
            fn_kwargs={'tokenizer': tokenizer, 'model_type': model_type}
        )
        tokenized_dataset = tokenized_dataset.map(
            recombine_text,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
        )
    return tokenized_dataset


def tokenize_data(examples, tokenizer, model_type, max_length=1000):
    """Tokenize an example.

    This method does tokenizing of the example depending on model type.

    Arguments
        examples: an element of Dataset object.
        tokenizer: tokenizer object.
        model_type: type of model to load. Either question answering ('qa') 
                    or causal language modelling ('lm') framework.
        max_length: max_length to use in tokenizing input for truncation and padding.

    Returns
    -------
        Returns tokenized element of Dataset object.
    """
    if model_type == 'qa':
        inputs = examples['question']
        targets = examples['answer']
        model_inputs = tokenizer(inputs, truncation=True, max_length=max_length) # padding=True
        labels = tokenizer(targets, truncation=True, max_length=max_length) # padding=True
        model_inputs["labels"] = labels["input_ids"]
    elif model_type == 'lm':
        model_inputs = tokenizer(examples['text'])
    else: 
        raise ValueError("Invalid model type! Should be either for question answering or simple text generation.")
    return model_inputs


def recombine_text(examples):
    """Recombine text into new chunks for LM."""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // _BLOCK_SIZE) * _BLOCK_SIZE
    result = {
	k: [t[i : i + _BLOCK_SIZE] for i in range(0, total_length, _BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def training(model, tokenizer, dataset_processed, 
    model_type, output_dir, save_dir,
    batch_size=8, num_epochs=3, lr=5e-5, max_length=1000):
    """Run training.

    This method creates training config and data collators to late actually run the training.

    Arguments
        model: model object.
        tokenizer: tokenizer object.
        dataset_processed: dataset object.
        model_type: type of model to load. Either question answering ('qa') 
                    or causal language modelling ('lm') framework.
        output_dir: directory to save some outputs during training.
        save_dir: directory to save trained model after training.
        batch_size: batch size for training.
        num_epochs: number of epochs for training.
        lr: learning rate for training.
        max_length: max_length to use in tokenizing input for truncation and padding.
    """
    kwargs_config = {        
        'output_dir': output_dir,
        'overwrite_output_dir': True,
        'per_device_train_batch_size': batch_size,
        'gradient_checkpointing': True,
        'gradient_accumulation_steps': 4,
        'optim': "adafactor",
        'logging_steps': 0.5,
        'save_strategy': 'epoch',
        'weight_decay': 0.01,
        'learning_rate': lr,
        'evaluation_strategy': 'epoch',
        #'eval_steps': 400,
        #'tf32': True,
        'per_device_eval_batch_size': batch_size,
        #'evaluation_strategy': "epoch",
        'save_total_limit': 3,
        'num_train_epochs': num_epochs,
        #'predict_with_generate': True,
        #'fp16': True,
        #'fp16_full_eval': True,
    }
    if model_type == 'qa':
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='max_length', max_length=max_length)
        training_args = Seq2SeqTrainingArguments(**kwargs_config)
    elif model_type == 'lm':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(**kwargs_config)
    else: 
        raise ValueError("Invalid model type! Should be either for question answering or simple text generation.")
    kwargs_train = {
        'model': model,
        'args': training_args,
        'train_dataset': dataset_processed["train"],
        'eval_dataset': dataset_processed["validation"],
        'tokenizer': tokenizer,
        'data_collator': data_collator
    }    
    if model_type == 'qa':
        trainer = Seq2SeqTrainer(**kwargs_train)
    elif model_type == 'lm':
        trainer = Trainer(**kwargs_train)
    else: 
        raise ValueError("Invalid model type! Should be either for question answering or simple text generation.")        
    trainer.train()
    trainer.save_model(save_dir)


def finetune(model_name, path_dataset, train_outputs, batch_size, num_epochs, lr):
    """"Sequence of steps to finetune a model."""
    dataset, tokenizer, model, model_type = preload(model_name, path_dataset)
    max_length = max_length_est(dataset, tokenizer, model_type)
    dataset_processed = preprocess(dataset, tokenizer, model_type, max_length=max_length)
    training(model, tokenizer, dataset_processed, model_type, 
        output_dir=train_outputs[0], save_dir=train_outputs[1], 
        batch_size=batch_size, num_epochs=num_epochs, lr=lr, max_length=max_length)
