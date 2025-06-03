import json

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from datasets import Dataset
from litellm import Router
from openai import OpenAI

from utils import parse_answer, parse_feedback


def openai_batch(client, messages_batch, dataset, round_idx, operation, time, gen_config):
    batch_file_name="./tmp_batch/dr_%s_%s_r%d_%s.jsonl" % (dataset, time, round_idx, operation)
    with open(batch_file_name, mode='w', encoding="utf-8") as f_batch:
        for i, messages in messages_batch:
            temp_request={"custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", "body": {"messages": messages, **gen_config}}
            f_batch.write(json.dumps(temp_request, ensure_ascii=False)+'\n')

    batch_input_file=client.files.create(file=open(batch_file_name, mode="rb"), purpose="batch")
    batch_input_file_id=batch_input_file.id
    batch_running=client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": batch_file_name}
    )

    batch_running_id=batch_running.id
    return batch_running_id

def claude_batch(client, messages_batch, gen_config):
    requests=[]
    for i, messages in messages_batch:
        requests.append(Request(custom_id=str(i), params=MessageCreateParamsNonStreaming(messages=messages, **gen_config)))

    batch_running=client.messages.batches.create(requests=requests)

    batch_running_id=batch_running.id
    return batch_running_id

def gemini_batch(messages_batch, dataset, gen_config, num_proc):
    messages_batch_hf=[{"id": i, "messages": messages} for i, messages in messages_batch]
    tmp_dataset=Dataset.from_list(messages_batch_hf)

    litellm_model_list=[{
        "model_name": "gemini-1-5",
        "litellm_params": {
            "model": gen_config["model"],
            "api_key": gen_config["api_key"],
            "base_url": gen_config["base_url"]
        }
    }]

    def get_answer_from_api(example):
        router=Router(model_list=litellm_model_list, num_retries=5, retry_after=10)
        messages=example["messages"]

        api_done=False
        while not api_done:
            try:
                response=router.completion(messages=messages, model="gemini-1-5", temperature=gen_config["temperature"], max_tokens=gen_config["max_tokens"])
                api_done=True
            except:
                api_done=False
        tmp_answer, tmp_raw=parse_answer(response.choices, dataset)

        if hasattr(response.usage.prompt_tokens_details, "cached_tokens"):
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens=0

        return {"answer": tmp_answer["answer"], "reasoning": tmp_answer["reasoning"], "raw": tmp_raw, "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "cached_tokens": cached_tokens}
    
    answered_dateset=tmp_dataset.map(get_answer_from_api, load_from_cache_file=False, num_proc=num_proc)

    return answered_dateset

def gemini_batch_feedback(messages_batch, gen_config, num_proc):
    messages_batch_hf=[{"id": i, "messages": messages} for i, messages in messages_batch]
    tmp_dataset=Dataset.from_list(messages_batch_hf)

    litellm_model_list=[{
        "model_name": "gemini-1-5",
        "litellm_params": {
            "model": gen_config["model"],
            "api_key": gen_config["api_key"],
            "base_url": gen_config["base_url"]
        }
    }]

    def get_answer_from_api(example):
        router=Router(model_list=litellm_model_list, num_retries=5, retry_after=10)
        messages=example["messages"]

        api_done=False
        while not api_done:
            try:
                response=router.completion(messages=messages, model="gemini-1-5", temperature=gen_config["temperature"], max_tokens=gen_config["max_tokens"])
                api_done=True
            except:
                api_done=False
        tmp_feedback, tmp_raw=parse_feedback(response.choices)

        if hasattr(response.usage.prompt_tokens_details, "cached_tokens"):
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens=0

        return {"feedback": tmp_feedback, "raw": tmp_raw, "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "cached_tokens": cached_tokens}
    
    answered_dateset=tmp_dataset.map(get_answer_from_api, load_from_cache_file=False, num_proc=num_proc)

    return answered_dateset

def vllm_batch(messages_batch, dataset, gen_config, num_proc, world_size):
    messages_batch_hf=[{"id": i, "messages": messages} for i, messages in messages_batch]
    tmp_dataset=Dataset.from_list(messages_batch_hf)

    def get_answer_from_api(example, rank):
        if not rank:
            rank=0
        clients=[OpenAI(api_key="EMPTY", base_url=("http://localhost:%d/v1" % (58000+w))) for w in range(world_size)]
        messages=example["messages"]

        # balance different gpus according to the proc rank.
        response=clients[rank%world_size].chat.completions.create(messages=messages, **gen_config)
        tmp_answer, tmp_raw=parse_answer(response.choices, dataset)

        if hasattr(response.usage.prompt_tokens_details, "cached_tokens"):
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens=0

        return {"answer": tmp_answer["answer"], "reasoning": tmp_answer["reasoning"], "raw": tmp_raw, "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "cached_tokens": cached_tokens}
    
    answered_dateset=tmp_dataset.map(get_answer_from_api, load_from_cache_file=False, num_proc=num_proc, with_rank=True)

    return answered_dateset

def vllm_batch_feedback(messages_batch, gen_config, num_proc, world_size):
    messages_batch_hf=[{"id": i, "messages": messages} for i, messages in messages_batch]
    tmp_dataset=Dataset.from_list(messages_batch_hf)

    def get_answer_from_api(example, rank):
        if not rank:
            rank=0
        clients=[OpenAI(api_key="EMPTY", base_url=("http://localhost:%d/v1" % (58000+w))) for w in range(world_size)]
        messages=example["messages"]

        response=clients[rank%world_size].chat.completions.create(messages=messages, **gen_config)
        tmp_feedback, tmp_raw=parse_feedback(response.choices)

        if hasattr(response.usage.prompt_tokens_details, "cached_tokens"):
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens=0

        return {"feedback": tmp_feedback, "raw": tmp_raw, "prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "cached_tokens": cached_tokens}
    
    answered_dateset=tmp_dataset.map(get_answer_from_api, load_from_cache_file=False, num_proc=num_proc, with_rank=True)

    return answered_dateset
