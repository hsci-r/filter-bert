# Based on the HuggingFace tutorial: "Fine-tuning a masked language model"
# https://huggingface.co/learn/nlp-course/chapter7/3

import argparse
import csv
import math
import pyarrow
import pyarrow.csv
from datasets import Dataset
from transformers import \
    AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, \
    TrainingArguments, Trainer


def tokenize_function(examples):
    result = tokenizer(examples['text'], is_split_into_words=True)
    if tokenizer.is_fast:
        result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
    return result


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune BERT for folk poetry.')
    parser.add_argument(
        '-i', '--input-file', metavar='FILE',
        help='The input file (CSV) containing verses to train on.')
    parser.add_argument(
        '-m', '--model', metavar='MODEL',
        default='TurkuNLP/bert-base-finnish-uncased-v1',
        help='The model name to take as base.')
    parser.add_argument(
        '-o', '--output-dir', metavar='PATH',
        help='The directory to save the output model to.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    data = pyarrow.csv.read_csv(args.input_file)
    data_spl = Dataset(pyarrow.table(
        [pyarrow.compute.split_pattern(data['text'], '_')],
        names=['text']
    ))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    tokens = data_spl.map(tokenize_function, batched=True, remove_columns=['text'])
    chunk_size = 64
    lm_data = tokens.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15)
    
    # TODO ???
    train_size = 400000
    test_size = int(0.1 * train_size)
    
    downsampled_dataset = lm_data.train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    
    batch_size = 32
    # Show the training loss with every epoch
    model_name = args.model.split('/')[-1]
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=downsampled_dataset['train'],
        eval_dataset=downsampled_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity {math.exp(eval_results['eval_loss']):.2f}")

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

