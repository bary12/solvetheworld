"""
Agent for reproducing issues
"""

import json
import git
import os
import subprocess
import openai
import pty
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import torch
from datasets import Dataset
import re
from trl import GRPOConfig
from grpo_trainer import UnslothGRPOTrainer as GRPOTrainer
from vllm.outputs import RequestOutput, CompletionOutput
from vllm import LLM
from typing import List
import textwrap
import types
import uuid
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
    max_seq_length = 2048
    lora_rank = 64
    shell_args = ['/bin/bash']
else:
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    max_seq_length = 1024
    lora_rank = 8
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
        "issue": issue['issue_title'] + '\n\n' + issue['discussion'][0]['body']
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
    TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>')
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
            try:
                tool_call = json.loads(tool_call)
            except json.JSONDecodeError:
                continue
            if tool_call['name'] == 'run_shell_command':
                output = self.run_shell_command(tool_call['arguments']['command'])
                self.add_message({
                    "role": "tool",
                    "name": tool_call['name'],
                    "content": output,
                })
            elif tool_call['name'] == 'done':
                self.finish()
                break
    
    def finish(self):
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/{uuid.uuid4()}.json', 'w') as f:
            json.dump(self.messages, f, indent=2)
        self.done = True


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
                    agents[i].finish()
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
    torch_compile = False, # for now don't compile, speeds up short training runs where inference is the bottleneck.
)

def _reward_for_done(prompt, completion, **kwargs):
    # parse all tool calls in completion
    tool_calls = Agent.TOOL_CALL_PATTERN.findall(completion)
    for tool_call in tool_calls:
        try:
            tool_call = json.loads(tool_call)
        except json.JSONDecodeError:
            continue
        if tool_call['name'] == 'done':
            return 1.
    return 0.

def reward_for_done(prompts, completions, **kwargs):
    return [_reward_for_done(prompt, completion) for prompt, completion in zip(prompts, completions)]

def _reward_for_reproducing_issue(prompt, completion, issue, **kwargs):
    # parse all tool calls and tool responses
    tool_call_matches = Agent.TOOL_CALL_PATTERN.findall(completion)
    tool_calls = []
    for tool_call_match in tool_call_matches:
        try:
            tool_call = json.loads(tool_call_match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    # parse all the tool responses
    TOOL_RESPONSE_PATTERN = re.compile(r'<tool_response>(.*?)</tool_response>')
    tool_response_matches = TOOL_RESPONSE_PATTERN.findall(completion)
    tool_responses = []
    for tool_response_match in tool_response_matches:
        tool_responses.append(tool_response_match)

    # prepare for llm
    session = ''
    for tool_call, tool_response in zip(tool_calls, tool_responses):
        if tool_call['name'] != 'run_shell_command':
            continue
        session += f'$ {tool_call["arguments"]["command"]}\n{tool_response}\n\n'
    
    if not any(tool_call['name'] == 'done' for tool_call in tool_calls):
        return 0.
    
    # run the llm
    response = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You will be given a description of an issue, along with a shell session that attempts to reproduce the issue. You will need to determine if the issue was reproduced successfully. respond with a json object like {\"success\": true} if the issue was reproduced successfully, or {\"success\": false} if it was not."},
            {"role": "user", "content": issue + '\n\nReproduction attempt:\n\n' + session},
        ],
        response_format={
            "type": "json_object",
            "json_schema": {
                "name": "success",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                        },
                    },
                },
            },
        },
    )
    return 1. if json.loads(response.choices[0].message.content)['success'] else 0.


def reward_for_reproducing_issue(prompts, completions, **kwargs):
    issues = kwargs['issue']
    repro = [_reward_for_reproducing_issue(prompt, completion, issue=issue) for prompt, completion, issue in zip(prompts, completions, issues)]
    print(repro)
    return repro

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [reward_for_done, reward_for_reproducing_issue],
    args = training_args,
    train_dataset = dataset,
)

trainer.llm = LLMWrapper(trainer.llm)

if __name__ == '__main__':
    trainer.train()