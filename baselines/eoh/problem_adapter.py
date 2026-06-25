import logging
import os
import subprocess
import re

from baselines.eoh.gls_tsp_adapt.gls_tsp_eval import Sandbox
from utils.utils import block_until_running, file_to_string, filter_traceback

class Prompts:
    def __init__(self, problem_cfg, root_dir:str):
        self.cfg = problem_cfg
        self.problem = problem_cfg.problem_name
        self.root_dir = root_dir
        self.problem_type = problem_cfg.problem_type
        self.prompt_dir = f"{self.root_dir}/prompts"

        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'

        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt').format(version=2).strip()
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')

        match = re.match(r'^def +(.+?_v2)\((.*)\) *-> *(.*?) *:', self.func_signature)
        assert match is not None
        self.prompt_func_name = match.group(1)
        self.prompt_func_inputs = [txt.split(":")[0].strip() for txt in match.group(2).split(",")]
        if self.prompt_func_name.startswith('select_next_node'):
            self.prompt_func_outputs = ['next_node']
        elif self.prompt_func_name.startswith('priority'):
            self.prompt_func_outputs = ['priorities']
        elif self.prompt_func_name.startswith('heuristics'):
            self.prompt_func_outputs = ['heuristics_matrix']
        elif self.prompt_func_name.startswith('crossover'):
            self.prompt_func_outputs = ['offsprings']
        else:
            self.prompt_func_outputs = ['result']

    def get_task(self):
        return self.cfg.description
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.func_desc

    def get_other_inf(self):
        return ""

class Problem:
    def __init__(self, cfg, root_dir):
        self.config = cfg
        self.root_dir = root_dir

        self.problem = self.config.problem.problem_name
        self.problem_description = self.config.problem.description
        self.problem_size = self.config.problem.problem_size
        self.obj_type = self.config.problem.obj_type
        self.problem_type = self.config.problem.problem_type
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        if self.problem_type == "tsp_constructive":
            from .original.prompts.tsp_greedy import GetPrompts
            self.prompts = GetPrompts()
        elif self.problem_type == "online":
            from .original.prompts.bpp_online import GetPrompts
            self.prompts = GetPrompts()
        else:
            print(self.problem_type)
            self.prompts = Prompts(self.config.problem, root_dir)

    def response_to_individual(self, code, response_id, file_name=None) -> dict:
        """
        Convert response to individual
        """
        outdir = './evaluations/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        runid = hash(code)
        # Write response to file
        file_name = outdir+f"problem_eval{runid}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(code + '\n')

        # Extract code and description from response
        std_out_filepath = outdir+f"problem_eval{runid}_stdout.txt" if file_name is None else file_name.rstrip(".txt") + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": outdir+f"problem_eval{runid}_code.py",
            "code": code,
            "response_id": response_id,
        }
        return individual
    
    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual


    def batch_evaluate(self, codes: list[str], iteration: int) -> list[dict]:
        """
        Evaluate population by running code in parallel and computing objective values and fitness.
        """
        self.iteration = iteration
        population = [self.response_to_individual(resp, index) for index, resp in enumerate(codes)]
        inner_runs = []

        for response_id, individual in enumerate(population):
            runid = hash(individual["code"])

            if individual["code"] is None:
                population[response_id] = self.mark_invalid_individual(individual, "Invalid response!")
                inner_runs.append(None)
                continue

            logging.info(f"Iteration {self.iteration}: Running Code {runid}")

            try:
                logging.debug(f"Iteration {self.iteration}: Processing Code Run {runid}")

                if self.problem != 'tsp_gls':
                    with open(self.output_file, 'w') as file:
                        file.writelines(individual["code"] + '\n')

                    stdout_filepath = individual["stdout_filepath"]
                    file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py'

                    with open(stdout_filepath, 'w') as f:
                        process = subprocess.Popen(
                            ['python', '-u', file_path, f'{self.problem_size}', self.root_dir, "train"], stdout=f,
                            stderr=f)

                    block_until_running(stdout_filepath, log_status=True)
                    inner_runs.append(process)

                else:
                    # Special case handling for tsp_gls
                    sand_box = Sandbox()
                    result, run_ok = sand_box.run(individual["code"])

                    if not run_ok:
                        raise RuntimeError("Sandbox execution failed")

                    # Assuming `result` contains the output that you would otherwise read from the stdout file
                    stdout_str = result
                    individual["stdout_filepath"] = "Sandbox Execution"

                    # Process the result directly, similar to how it's done after reading the stdout file
                    traceback_msg = filter_traceback(stdout_str)

                    if not traceback_msg:
                        try:
                            individual["obj"] = float(stdout_str.split('\n')[-2])
                            assert individual["obj"] > 0, "Objective value <= 0 is not supported."
                            if self.obj_type == "max":
                                individual["obj"] = -individual["obj"]
                            individual["exec_success"] = True
                        except Exception as e:
                            population[response_id] = self.mark_invalid_individual(individual,
                                                                                   "Invalid stdout / objective value!")
                            logging.error(f"Error processing objective for response_id {response_id}: {e}")
                    else:
                        population[response_id] = self.mark_invalid_individual(individual, traceback_msg)
                        logging.error(f"Traceback for response_id {response_id}: {traceback_msg}")

                    logging.info(
                        f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual.get('obj')}")
                    inner_runs.append(None)  # No process to manage for Sandbox

            except Exception as e:
                logging.error(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(individual, str(e))
                inner_runs.append(None)
                continue

        # Wait for all processes to finish and collect results for non-Sandbox evaluations
        for response_id, process in enumerate(inner_runs):
            if process is None:
                continue

            try:
                process.communicate(timeout=self.config.timeout)
            except subprocess.TimeoutExpired as e:
                logging.error(f"Timeout for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                process.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]

            try:
                with open(stdout_filepath, 'r') as f:
                    stdout_str = f.read()

                traceback_msg = filter_traceback(stdout_str)

                if not traceback_msg:  # If execution has no error
                    try:
                        individual["obj"] = float(stdout_str.split('\n')[-2])
                        assert individual["obj"] > 0, "Objective value <= 0 is not supported."
                        if self.obj_type == "max":
                            individual["obj"] = -individual["obj"]
                        individual["exec_success"] = True
                    except Exception as e:
                        population[response_id] = self.mark_invalid_individual(individual,
                                                                               "Invalid stdout / objective value!")
                        logging.error(f"Error processing objective for response_id {response_id}: {e}")
                else:
                    population[response_id] = self.mark_invalid_individual(individual, traceback_msg)
                    logging.error(f"Traceback for response_id {response_id}: {traceback_msg}")

                logging.info(
                    f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual.get('obj')}")

            except Exception as e:
                logging.error(f"Failed to read stdout for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(individual, "Failed to read stdout")

        return [indiv.get("obj") for indiv in population]
