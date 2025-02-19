"""
Agent for reproducing issues
"""

import json
import git
import os
import subprocess
import pty
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import torch
from datasets import Dataset
import re
from trl import GRPOConfig
from grpo_trainer import GRPOTrainer
from vllm.outputs import RequestOutput, CompletionOutput
from vllm import LLM
from typing import List
import textwrap
import types

PatchFastRL("GRPO", FastLanguageModel)

try:
  import google.colab
  IS_COLAB = True
except:
  IS_COLAB = False

with open('projects.json', 'r') as f:
    projects = json.load(f)

if IS_COLAB:
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    max_seq_length = 8192
    lora_rank = 64
    shell_args = ['/bin/bash']
else:
    model_name = 'Qwen/Qwen2.5-3B-Instruct'
    max_seq_length = 512
    lora_rank = 32
    container_name = 'repro-agent'
    shell_args = ['docker', 'exec', '-it', container_name, '/bin/bash']

workspaces_dir = 'workspaces'
os.makedirs(workspaces_dir, exist_ok=True)
repos = {}

for project in projects['projects']:
    # project is a string like "owner/name", get the github url
    github_url = f"https://github.com/{project}"
    # clone the repo, if it does not exist
    repo_dir = os.path.join(workspaces_dir, project)
    if not os.path.exists(repo_dir):
        git.Repo.clone_from(github_url, repo_dir)
    # pull the latest changes
    repo = git.Repo(repo_dir)
    repos[project] = repo


with open('issues.json', 'r') as f:
    issues = json.load(f)['issues']

UNIQUE_SEPARATOR = '__dehydrate_the_masses__'

dataset = Dataset.from_list([
    {
        # Since GRPOTrainer only passes "prompt" into the vllm model we later wrap,
        # We use an absolutely horrible hack to pass the metadata to the model.
        "prompt": UNIQUE_SEPARATOR.join([
            (issue['issue_title'] + '\n\n' + issue['discussion'][0]['body']),
            issue['repository'],
            issue['merged_prs'][0]['base_commit']['sha']
        ]),
    }
    for issue in issues
])

class Shell:
    def __init__(self, cwd):
        self.master_fd, self.slave_fd = pty.openpty()
        kwargs = {}
        if IS_COLAB:
            kwargs['cwd'] = cwd
        self.process = subprocess.Popen(
            shell_args + ['--noprofile', '--norc'],
            stdin=subprocess.PIPE,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            text=True,
            close_fds=True,
            env={
                "PS1": UNIQUE_SEPARATOR,
            }
            **kwargs
        )
        self.master = None

        os.close(self.slave_fd)

    def __enter__(self):
        self.master = os.fdopen(self.master_fd, 'r+b')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.master.close()
        self.process.terminate()

    def run_command(self, command):
        if self.master is None:
            raise RuntimeError("Shell should be called using the with statement")

        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

        # read one line at a time, until we see the UNIQUE_SEPARATOR
        lines = []
        while True:
            line = self.process.stdout.readline()
            if line.strip().endswith(UNIQUE_SEPARATOR):
                break
            lines.append(line.strip())
        return '\n'.join(lines)

class Agent:
    TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>')
    def __init__(self, prompt, repo, commit_hash):
        self.messages = [
            {"role": "system", "content": "You will be given a description of an issue with the project, you will need to reproduce the issue. You can explore the codebase using the available tools."},
            {"role": "user", "content": prompt},
        ]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Run a shell command. You can use cat, ls and grep to explore the codebase.",
                    "parameters": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run.",
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "done",
                    "description": "Call this tool when you have reproduced the issue.",
                    "parameters": {},
                }
            }
        ]
        self.repo = repo
        self.commit_hash = commit_hash
        self.done = False

    def add_message(self, message):
        self.messages.append(message)

    def apply_chat_template(self):
        return tokenizer.apply_chat_template(
            self.messages,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=False,
        )

    def as_completion_output(self):
        chat_template = self.apply_chat_template()
        token_ids = tokenizer.encode(chat_template)
        return CompletionOutput(
            index=0,
            text=chat_template,
            token_ids=token_ids,
            logprobs=None,
            cumulative_logprob=None,
        )

    def parse_response(self, response):
        tool_calls = self.TOOL_CALL_PATTERN.findall(response.text)
        rest_of_message = re.sub(self.TOOL_CALL_PATTERN, '', response.text)
        self.add_message({
            "role": "assistant",
            "content": rest_of_message,
        })
        # the group(1) is a json string in format {"name": "name", "arguments": {"arg": "value"}}
        for tool_call in tool_calls:
            tool_call = json.loads(tool_call)
            if tool_call['name'] == 'run_shell_command':
                output = self.run_shell_command(tool_call['arguments']['command'])
                self.add_message({
                    "role": "tool",
                    "name": tool_call['name'],
                    "content": output,
                })
            elif tool_call['name'] == 'done':
                self.done = True
                break
        


class LLMWrapper:
    """
    We monkey-patch this class into GRPOTrainer so that we can control the parameters to the vllm call.
    """
    def __init__(self, model):
        self.model: LLM = model

    def generate(self, all_prompts_text, *args, **kwargs):
        """ main agent loop """
        # Resolve the hack from earlier
        sampling_params = kwargs['sampling_params']
        batch_size = len(all_prompts_text)
        generations_per_prompt = sampling_params.n
        unzipped = [x.split(UNIQUE_SEPARATOR) for x in all_prompts_text]
        prompts = [x[0] for x in unzipped]
        repos_ = [repos[x[1]] for x in unzipped]
        commit_hashes = [x[2] for x in unzipped]
        agents = [
            [Agent(prompt, repo, commit_hash) for _ in range(generations_per_prompt)]
            for prompt, repo, commit_hash in zip(prompts, repos_, commit_hashes)
        ]
        # flatten the list of agents
        agents = [agent for sublist in agents for agent in sublist]
        sampling_params.n = 1

        agent_outputs: List[CompletionOutput] = [None] * len(agents)

        while not all(agent.done for agent in agents):
            not_done_agents = [(i, agent) for i, agent in enumerate(agents) if not agent.done]
            ids, not_done_agents = zip(*not_done_agents)
            prompts = [agent.apply_chat_template() for agent in not_done_agents]
            generation = self.model.generate(prompts, *args, **kwargs)
            outputs = generation[0].outputs
            for i, output in zip(ids, outputs):
                if output.finish_reason == "length":
                    agents[i].done = True
                else:
                    agents[i].parse_response(output)
                

        # return in the format GRPOTrainer expects
        ret = [
            RequestOutput(
                request_id=None,
                outputs=[
                    agents[i * generations_per_prompt + j].as_completion_output()
                    for j in range(generations_per_prompt)
                ],
                prompt=None,
                prompt_token_ids=None,
                prompt_logprobs=None,
                finished=True,
            )
            for i in range(batch_size)
        ]
        breakpoint()
        return ret


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 1024,  # shorter prompts will be truncated, I guess?
    max_completion_length = max_seq_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "wandb",
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [lambda x: 0],  # for now just return 0
    args = training_args,
    train_dataset = dataset,
)

trainer.llm = LLMWrapper(trainer.llm)

def monkey_patch_grpotrainer_to_not_concat_prompts():
    import trl.trainer.grpo_trainer as grpo_trainer_module
    source_lines, _ = inspect.getsourcelines(grpo_trainer_module.GRPOTrainer._prepare_inputs)
    source_lines = [
        line.replace('prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)', 'prompt_completion_ids = completion_ids') \
        .replace('prompt_completion_ids = completion_ids', 'prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)')
        for line in source_lines
    ]

trainer.train()