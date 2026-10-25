"""Collect safe-informative GPT-4 responses for router training (Appendix B).

GPT-4 answers the harmful PKU-SafeRLHF prompts of the safe-data categories
(Drug Abuse & Weapons, Psychological Harm, Cybercrime, Economic Crime) with
the prompt below; blanket refusals are resampled up to --max_attempts times
and prompts that only ever get refusals are skipped.

    python scripts/data/generate_safe_responses.py --output data/safe_raw.json --per_category 200

The paper used gpt-4; OpenAI has since retired it, so pass a current model with
--model (e.g. gpt-4o). Then run scripts/data/clean_safe_responses.py on the output. Needs OPENAI_API_KEY
in the environment. Output: {"prompt", "response", "category"} records; reruns resume.
"""

import argparse
import os
import time

import openai

from safemoe.constants import SAFE_DATA_CATEGORIES
from safemoe.data import load_json, save_json
from safemoe.datasets import load_records
from safemoe.refusals import clean_response

PROMPT = "Respond with some reasoning to the following questions rather than simple rejections: {prompt}"


def complete(client, model, prompt, temperature, retries=4):
    """One GPT-4 answer; API errors are retried and finally raised, never counted as refusals."""
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT.format(prompt=prompt)}],
                temperature=temperature,
            ).choices[0].message.content
        except openai.OpenAIError as e:
            if attempt == retries - 1:
                raise
            print(f"API error ({e}); retrying")
            time.sleep(10 * 2 ** attempt)


def ask(client, model, prompt, max_attempts, temperature):
    """Resample until the answer is not a blanket refusal; None if it always is."""
    for _ in range(max_attempts):
        reply = complete(client, model, prompt, temperature)
        if clean_response(reply):  # something informative is left once refusals are removed
            return reply
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--categories", nargs="+", default=SAFE_DATA_CATEGORIES)
    p.add_argument("--split", default="train")
    p.add_argument("--per_category", type=int, default=None, help="stop once this many are kept (default: all prompts)")
    p.add_argument("--model", default="gpt-4", help="the paper's model (retired by OpenAI; override it)")
    p.add_argument("--max_attempts", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    client = openai.OpenAI()
    done = load_json(args.output) if os.path.exists(args.output) else []
    seen = {r["prompt"] for r in done}
    for category in args.categories:
        kept = sum(r["category"] == category for r in done)
        prompts = dict.fromkeys(r["prompt"] for r in load_records("pku", split=args.split, category=category))
        for prompt in prompts:
            if args.per_category is not None and kept >= args.per_category:
                break
            if prompt in seen:
                continue
            seen.add(prompt)
            reply = ask(client, args.model, prompt, args.max_attempts, args.temperature)
            if reply is not None:
                done.append({"prompt": prompt, "response": reply, "category": category})
                kept += 1
                if kept % 20 == 0:
                    save_json(done, args.output)
        save_json(done, args.output)
        print(f"{category}: {kept} safe-informative responses")


if __name__ == "__main__":
    main()
