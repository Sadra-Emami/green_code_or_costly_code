import asyncio
import json
import os
import copy
import argparse
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
from datasets import load_dataset


def param_suffix(max_tokens: int, temperature: float, context_size: int | None) -> str:
    """Build a '_t0.2_mt4096_ctx16384' style suffix for non-default params."""
    parts = []
    if temperature != 0.0:
        parts.append(f"t{temperature:g}")
    if max_tokens != 2048:
        parts.append(f"mt{max_tokens}")
    if context_size is not None:
        parts.append(f"ctx{context_size}")
    return ("_" + "_".join(parts)) if parts else ""


def build_prompt(data_entry: dict) -> str:
    if data_entry["dataset"] == "EffiBench":
        test_case = data_entry["small_test_cases"]
        return (
            f"Please complete Python code based on the task description and test cases. "
            f"# Task description:\n{data_entry['markdown_description']}\n{test_case}\n#Solution:\n"
        )
    elif data_entry["dataset"] == "HumanEval":
        return (
            f"Please complete Python code based on the task description. "
            f"# Task description:\n{data_entry['prompt']}\n#Solution:\n"
        )
    elif data_entry["dataset"] == "MBPP":
        tests = "\n".join(data_entry["test_list"])
        return (
            f"Please complete Python code based on the task description and test cases. "
            f"# Task description:\n{data_entry['prompt']}\n{tests}\n#Solution:\n"
        )
    return ""


async def fetch_completion(
    data_entry: dict,
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> dict:
    prompt = build_prompt(data_entry)
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            if not content:
                print("  [WARN] Empty content.")
            data_entry["completion"] = content
        except Exception as e:
            print(repr(e))
            data_entry["completion"] = "Error"
    return data_entry


async def main_async(args):
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc123",  # vLLM ignores the key; just needs a non-empty value
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    if args.dataset == "EffiBench":
        with open("../datasets/dataset.json", "r") as f:
            dataset = json.load(f)
    elif args.dataset == "HumanEval":
        dataset = list(load_dataset("evalplus/humanevalplus", split="test"))
    elif args.dataset == "MBPP":
        dataset = list(load_dataset("evalplus/mbppplus", split="test"))

    for entry in dataset:
        entry["dataset"] = args.dataset

    if args.limit is not None:
        dataset = dataset[: args.limit]

    print(f"Sending {len(dataset)} requests to vLLM (concurrency={args.concurrency}) ...")

    tasks = [
        fetch_completion(copy.deepcopy(entry), args.checkpoint, client, semaphore,
                         args.max_tokens, args.temperature)
        for entry in dataset
    ]
    results = await async_tqdm.gather(*tasks)

    base_name = args.model_name if args.model_name else args.checkpoint.replace("/", "-").replace(":", "-")
    model_name = base_name + param_suffix(args.max_tokens, args.temperature, args.context_size)
    results_dir = f"../results/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"{args.dataset}_vllm_{model_name}.json")
    with open(out_file, "w") as f:
        json.dump(list(results), f, indent=4)
    print(f"Saved {len(results)} entries to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Model name as served by vLLM (e.g. Qwen/Qwen2.5-Coder-32B-Instruct)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["EffiBench", "HumanEval", "MBPP"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max number of in-flight requests to vLLM at once")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override output file name (default: derived from --checkpoint)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of tasks (for testing)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max output tokens per completion (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--context-size", type=int, default=None,
                        help="Label only: the --max-model-len set on the vLLM server. "
                        "Included in the output folder/filename suffix for tracking.")
    args = parser.parse_args()

    asyncio.run(main_async(args))
