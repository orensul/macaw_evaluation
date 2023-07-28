The folder "GPT_choose_dims_generate_elaborations" includes a notebook to generate dimensions and elaborations via GPT-3.5 on given datasets.

The folder "MacawProject" contains the files related to the macaw model as well as
files related to the datasets and the prompts to GPT-3.5

In addition, it includes python files for evaluation of Macaw.

"eval_macaw.py" -- evaluation of macaw models with context (dimensions + elaborations) or without context.
it reads the input test data files for evaluation from the folders "test_sets_100samples"  
and generates output files in the folders of "gpt3_eval_results" and "no_contexts_eval_results"

"macaw_eval_results.py" just prints the results of evaluation by reading the output evaluation files from
folders: "gpt3_eval_results" and "no_contexts_eval_results"
