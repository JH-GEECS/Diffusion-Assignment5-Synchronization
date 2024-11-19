import ray
import json
import argparse

import os
import subprocess
import signal

from typing import List, Dict, Union

def load_json(path: str)-> List[Dict[str, str]]:
    with open(path, "r") as f:
        data = json.load(f)
    
    if "wide" in path:
        ret_list = []
        for k, v in data.items():
            ret_list.append({"wide_prompt": v["prompt"]})
    elif "ambiguous" in path:
        ret_list = []
        for k, v in data.items():
            ret_list.append({"canonical_prompt": v["canonical_prompt"],
                             "instance_prompt": v["instance_prompt"]})
    else:
        raise ValueError("Invalid path")
    
    return ret_list

# @ray.remote
def cli_runner(input: dict, cli_args: argparse.Namespace):
    # 자식 프로세스를 추적하기 위한 리스트
    child_processes = []
    
    def handle_sigterm(signum, frame):
        print("SIGTERM received. Terminating child processes...")
        # 모든 자식 프로세스 종료
        for proc in child_processes:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        # 현재 프로세스 종료
        os._exit(0)
        
    if "wide_prompt" in input:
        wide_prompt = input["wide_prompt"]
        command = f"python main.py --app wide_image --prompt '{wide_prompt}' --tag wide_image --save_dir_now"        
    elif "canonical_prompt" in input:
        canonical_prompt = input["canonical_prompt"]
        instance_prompt = input["instance_prompt"]
        
        if cli_args.ambiguous_mode == "default":
            command = f"python main.py --app ambiguous_image --prompts '{canonical_prompt}' '{instance_prompt}' --tag ambiguous_image --save_dir_now"
        elif cli_args.ambiguous_mode == "inner_rot":
            command = f"python main.py --app ambiguous_image --prompts '{canonical_prompt}' '{instance_prompt}' --views_names identity inner_rotate  --tag ambiguous_image_inner_rotate_x_0  --save_dir_now"
        else:
            raise ValueError(f"Invalid ambiguous mode, current mode: {cli_args.ambiguous_mode}")
    else:
        raise ValueError("Invalid input")

    proc = subprocess.Popen(command, shell=True)
    child_processes.append(proc)
    proc.wait()
    
def main(cli_args):
    ray.init()
    
    wrapped_cli_runner = ray.remote(num_gpus=1)(cli_runner)
    ray_jobs = [] 
    count = 0
    for each_input in cli_args.input:
        jobs = load_json(each_input)
        for each_jobs in jobs:
            ray_jobs.append(wrapped_cli_runner.remote(each_jobs, cli_args))
            count += 1

    print(f"Total {count} jobs submitted.")
    result = ray.get(ray_jobs)
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs='+', required=True, help="List of input strings")
    parser.add_argument("--ambiguous_mode", type=str, default="default")
    args = parser.parse_args()
    
    main(args)