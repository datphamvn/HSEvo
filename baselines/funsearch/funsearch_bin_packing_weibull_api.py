import http.client
import json
import multiprocessing
import time
from argparse import ArgumentParser
from typing import Collection, Any, Tuple
import pickle
import requests
import os, sys
import tiktoken
import logging

# sys.path.append('../')

from funsearch_impl import funsearch
from funsearch_impl import config
from funsearch_impl import sampler
from funsearch_impl import evaluator_accelerate
from funsearch_impl import evaluator
from funsearch_impl import code_manipulation

parser = ArgumentParser()
parser.add_argument('--run', type=int)
parser.add_argument('--config', type=str, default='run_runtime_llm_config.json')
parser.add_argument('--llm', type=str, default='gemini-1.5-flash')
parser.add_argument('--key', type=str)
args = parser.parse_args()


def _trim_preface_of_body(sample: str) -> str:
    """Trim the redundant descriptions/symbols/'def' declaration before the function body.
    Please see my comments in sampler.LLM (in sampler.py).
    Since the LLM used in this file is not a pure code completion LLM, this trim function is required.

    -Example sample (function & description generated by LLM):
    -------------------------------------
    This is the optimized function ...
    def priority_v2(...) -> ...:
        return ...
    This function aims to ...
    -------------------------------------
    -This function removes the description above the function's signature, and the function's signature.
    -The indent of the code is preserved.
    -Return of this function:
    -------------------------------------
        return ...
    This function aims to ...
    -------------------------------------
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
        # find the first 'def' statement in the given code
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    return sample


class LLMAPI(sampler.LLM):
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, samples_per_prompt: int, timeout=30, trim=True):
        super().__init__(samples_per_prompt)
        additional_prompt = ('Complete a different and more complex Python function. '
                             'Be creative and you can insert multiple if-else and for-loop in the code logic.'
                             'Only output the Python code, no descriptions.'
                             'Do not repeat priority_v0 in your response')
        self._additional_prompt = additional_prompt
        self._trim = trim
        self._timeout = timeout
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = 'gemini'

    def cal_usage_LLM(self, lst_prompt, lst_completion, encoding_name="cl100k_base"):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        for i in range(len(lst_prompt)):
            if 'gemini' in self.model:
                self.prompt_tokens += len(encoding.encode(lst_prompt[i][0] + " " + lst_prompt[i][1]))
            else:
                for message in lst_prompt[i]:
                    for key, value in message.items():
                        self.prompt_tokens += len(encoding.encode(value))

            self.completion_tokens += len(encoding.encode(lst_completion[i]))

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        prompt = '\n'.join([content, self._additional_prompt])
        while True:
            try:
                if args.key is None:
                    raise Exception("API key is require input!")

                if 'gemini' in args.llm:
                    conn = http.client.HTTPSConnection("generativelanguage.googleapis.com")
                    payload = json.dumps({
                        "contents": [{"parts": [{"text": prompt}]}]
                    })
                    headers = {
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST",
                                 "/v1beta/models/gemini-1.5-flash-latest:generateContent?key=" + args.key,
                                 payload, headers)
                    res = conn.getresponse()
                    data = res.read().decode("utf-8")
                    data = json.loads(data)
                    # print("Prompt:", prompt)
                    response = data['candidates'][0]['content']['parts'][0]['text']
                else:
                    conn = http.client.HTTPSConnection("api.openai.com", timeout=self._timeout)
                    payload = json.dumps({
                        "model": args.llm,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    })
                    headers = {
                        'Authorization': 'Bearer ' + args.key,
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/v1/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = res.read().decode("utf-8")
                    data = json.loads(data)
                    response = data['choices'][0]['message']['content']

                self.cal_usage_LLM([['', prompt]], [response])
                print(f"LLM usage: prompt_tokens = {self.prompt_tokens}, completion_tokens = {self.completion_tokens}")

                append_to_file(f'logs/run{args.run}/main.log', f"LLM usage: prompt_tokens = {self.prompt_tokens}, completion_tokens = {self.completion_tokens}")
                # print("Response:", response)
                # trim function
                if self._trim:
                    response = _trim_preface_of_body(response)

                return response
            except Exception:
                time.sleep(2)
                continue


class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated code. Implemented by RZ.

    RZ: Sandbox returns the 'score' of the program and:
    1) avoids the generated code to be harmful (accessing the internet, take up too much RAM).
    2) stops the execution of the code in time (avoid endless loop).
    """

    def __init__(self, verbose=False, numba_accelerate=True):
        """
        Args:
            verbose         : Print evaluate information.
            numba_accelerate: Use numba to accelerate the evaluation. It should be noted that not all numpy functions
                              support numba acceleration, such as np.piecewise().
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program: str,
            function_to_run: str,  # RZ: refers to the name of the function to run (e.g., 'evaluate')
            function_to_evolve: str,  # RZ: accelerate the code by decorating @numba.jit() on function_to_evolve.
            inputs: Any,  # refers to the dataset
            test_input: str,  # refers to the current instance
            timeout_seconds: int,
            # **kwargs  # RZ: add this
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded.

        RZ: If the generated code (generated by LLM) is executed successfully,
        the output of this function is the score of a given program.
        RZ: PLEASE NOTE THAT this SandBox is only designed for bin-packing problem.
        """
        dataset = inputs[test_input]

        try:
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._compile_and_run_function,
                args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
            )
            process.start()
            process.join(timeout=timeout_seconds)
            if process.is_alive():
                # if the process is not finished in time, we consider the program illegal
                process.terminate()
                process.join()
                results = None, False
            else:
                if not result_queue.empty():
                    results = result_queue.get_nowait()
                else:
                    results = None, False
            return results
        except:
            return None, False

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate,
                                  result_queue):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_name=function_to_evolve
                )
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program, all_globals_namespace)
            # get the pointer of 'function_to_run'
            function_to_run = all_globals_namespace[function_to_run]
            # return the execution results
            results = function_to_run(dataset)

            # the results must be int or float
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except:
            # if raise any exception, we assume the execution failed
            result_queue.put((None, False))


def append_to_file(file_path, text):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Append the text to the file, creating it if it doesn't exist
    with open(file_path, 'a') as file:
        file.write(text + '\n')

# It should be noted that the if __name__ == '__main__' is required.
# Because the inner code uses multiprocess evaluation.
if __name__ == '__main__':
    specification = r'''
import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items, bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)


@funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    ratios = item / bins
    log_ratios = np.log(ratios)
    priorities = -log_ratios
    return priorities
'''

    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=20)

    # if it is set to None, funsearch will execute an endless loop
    global_max_sample_num = 4

    # load dataset
    with open('bin_packing/weibull_datasets/weibull_5k_train.pkl', 'rb') as f:
        weibull_5k_train = pickle.load(f)
        lst_key_weibull_5k_train = list(weibull_5k_train.keys())

        append_to_file(f'logs/run{args.run}/main.log', f"l1_bound: {weibull_5k_train['l1_bound']}")

    with open('bin_packing/weibull_datasets/weibull_train.pkl', 'rb') as f:
        bin_packing_weibull_train = pickle.load(f)

        for idx, keys in enumerate(bin_packing_weibull_train["weibull_5k_train"].keys()):
            bin_packing_weibull_train["weibull_5k_train"][keys] = weibull_5k_train[lst_key_weibull_5k_train[idx]]


    funsearch.main(
        specification=specification,
        inputs=bin_packing_weibull_train,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=f'logs/run{args.run}'
    )