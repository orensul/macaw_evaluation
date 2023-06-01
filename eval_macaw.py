import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cache_directory = '/cs/labs/dshahaf/orens/huggingface'

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_size', type=int, default=1000,
                        help='choose how many samples to read')
    args = parser.parse_args()
    return args


def read_train_samples(filename, num_samples_to_read):
    data = []
    count_samples = 0
    with open(filename, 'r') as f:
        for l in f:
            count_samples += 1
            if count_samples == num_samples_to_read + 1:
                break
            line = json.loads(l)
            sample_id = line["id"]
            mcoptions = line["mcoptions"]
            question = line['question']
            context = line['context']
            answer = line['answer']
            d = {"sample_id": sample_id, "mc": mcoptions, "question": question,
                 "context": context, "answer": answer}
            data.append(d)
        return data


def call_macaw(tokenizer, model, sample, with_context):
    input_string = "$answer$ ; $mcoptions$ = " + sample['mc'] + " ; $question$ = " + sample['question']
    if with_context:
        input_string += " $context$ = " + sample['context']

    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
    output = model.generate(input_ids)
    macaw_answer = tokenizer.batch_decode(output, skip_special_tokens=True)
    return macaw_answer


def macaw_eval_samples(tokenizer, model, samples, with_context):
    count_correct_answers = 0
    for sample in samples:
        macaw_answer = call_macaw(tokenizer, model, sample, with_context=with_context)[0].split('=')[1].lstrip()
        gt_answer = sample['answer']
        if macaw_answer == gt_answer:
            count_correct_answers += 1
    return count_correct_answers


def main(args):
    num_samples_to_read = args.samples_size
    print("num samples to read : " + str(num_samples_to_read))

    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-3b")
    print("Finished to load macaw tokenizer succesfully")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-3b", cache_dir=cache_directory).to(device)
    print("Finished to load macaw model succesfully")

    samples = read_train_samples('ETHICS_train_samples.jsonl', num_samples_to_read)
    print("Finished to read train samples succesfully")

    print()
    print("--- macaw evaluation without context ---")
    print()

    count_correct_answers = macaw_eval_samples(tokenizer, model, samples, with_context=False)
    print("macaw correct answers without context: " + str(count_correct_answers) + " out of " +
          str(len(samples)) + " questions")
    print("accuracy without context = " + str(count_correct_answers / len(samples)))

    print()
    print("--- macaw evaluation with context ---")
    print()

    count_correct_answers = macaw_eval_samples(tokenizer, model, samples, with_context=True)
    print("macaw correct answers with context: " + str(count_correct_answers) + " out of " +
          str(len(samples)) + " questions")
    print("accuracy with context = " + str(count_correct_answers / len(samples)))


if __name__ == "__main__":
    args = read_args()
    main(args)