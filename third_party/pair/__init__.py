"""Minimal vendored copy of PAIR (https://github.com/patrickrchao/JailbreakingLLMs, MIT License).

Source commit 6379ef7. Only what scripts/eval/pair_attack.py needs is kept:
  system_prompts.py  attacker and judge system prompts, unmodified
  common.py          get_init_msg, process_target_response (unmodified) and the
                     locally patched extract_json used for the reported runs
The model wrappers (litellm, fastchat, jailbreakbench) are replaced by direct
OpenAI / Anthropic SDK calls in the script.
"""
