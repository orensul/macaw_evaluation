import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cache_directory = '/cs/labs/dshahaf/orens/huggingface'


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--samples_size', type=int, default=-1,
                        help='choose how many samples to read')

    parser.add_argument('--run_with_context', type=bool, default=False,
                        help='choose True to run with context (elaboration), otherwise choose False')

    parser.add_argument('--dataset_file_name', type=str, default="",
                        help='choose the dataset file name')

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
    run_with_context = args.run_with_context
    dataset_file_name = args.dataset_file_name

    print("num samples to read : " + str(num_samples_to_read))
    print("should run with context : " + str(run_with_context))
    print("dataset file name: " + str(dataset_file_name))

    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-3b")
    print("Finished to load macaw tokenizer successfully")

    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-3b", cache_dir=cache_directory).to(device)
    print("Finished to load macaw model successfully")

    samples = read_train_samples(dataset_file_name, num_samples_to_read)
    print("Finished to read train samples successfully")

    print()
    print("--- macaw evaluation")
    print()

    count_correct_answers = macaw_eval_samples(tokenizer, model, samples, with_context=run_with_context)
    print("macaw correct answers without context: " + str(count_correct_answers) + " out of " +
          str(len(samples)) + " questions")
    print("accuracy without context = " + str(count_correct_answers / len(samples)))



if __name__ == "__main__":
    args = read_args()
    main(args)