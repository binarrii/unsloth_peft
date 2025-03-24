import ast
import os
import re
import textwrap

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


_CGREEN, _CRED, _CEND = "\033[92m", "\033[91m", "\033[0m"

_client = OpenAI(base_url="http://10.252.25.239:11434/v1", api_key="sk-xxx")


def _unquot(s):
    try:
        return ast.literal_eval(s)
    except:  # noqa: E722
        return s


if __name__ == "__main__":
    _out_file = "sensitive_questions.txt.c"
    if os.path.exists(_out_file):
        os.remove(_out_file)

    prompt_template = textwrap.dedent("""
    - 以 `{}` 这些词汇或短语生成 **2** 个问题
    - 生成的问题中 **必须** 包含这些词汇或短语 (IMPORTANT!!!)
    - 2个问题分2行输出, 不要输出任何其他内容 (IMPORTANT!!!)
    """)

    def _process_line(line, i):
        words, c = line.strip().split('@@')
        words = _unquot(words)
        response = _client.chat.completions.create(
            model="huihui_ai/qwen2.5-abliterate:32b",
            messages=[
                {
                    "role": "user",
                    "content": prompt_template.format(words),
                }
            ],
        )
        answer = response.choices[0].message.content
        answer = answer.split("</think>")[-1]
        answer = [q.strip() for q in answer.split("\n") if q.strip()]
        results = []
        for j, ans in enumerate(answer):
            ans = re.sub(r'^\s*(\d+\.?|-)?\s*', '', ans)
            results.append((words, c, i, ans, j))
        return results
    
    def _resolve_futures(futures):
        with open(_out_file, "a") as out_file:
            for future in futures:
                for result in future.result():
                    words, c, i, ans, j = result
                    out_file.write(f"{ans}@@{c}\n")
                    if j == 0:
                        print(f"{_CRED}{i+1}. {words}{_CEND}")
                    print(f"{_CGREEN}{i+1}-{j+1}. {ans}{_CEND}")
                print("\n")

    _N = 10
    with open("sensitive_words.txt.c", "r") as in_file:
        with ThreadPoolExecutor(max_workers=_N) as executor:
            _futures = []
            for i, line in enumerate(in_file):
                _futures.append(executor.submit(_process_line, line, i))
                if len(_futures) == _N:
                    _resolve_futures(_futures)
                    _futures = []
            _resolve_futures(_futures)
