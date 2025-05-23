import os
import re
import textwrap

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


_CGREEN, _CRED, _CEND = "\033[92m", "\033[91m", "\033[0m"

_client = OpenAI(base_url="http://10.252.25.239:11434/v1", api_key="sk-xxx")


if __name__ == "__main__":
    _out_file = "sensitive_questions.txt.t"
    if os.path.exists(_out_file):
        os.remove(_out_file)

    prompt_template = textwrap.dedent("""
    - 在 **原意不变** 的前提下把这个句子变换一下说法: `{}`
    - 一定要保证原意不变 (IMPORTANT!!!)
    - 如果可能, 加上敬语, 使语气更委婉 (OPTIONAL)
    - 只要给出一种/一个回答即可 (IMPORTANT!!!)
    - 用纯中文回答, 不要输出任何其他内容 (IMPORTANT!!!)
    """)

    def _process_line(line, i):
        words = line.strip()
        response = _client.chat.completions.create(
            model="google/gemma3-27b-fp16",
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
            results.append((words, i, ans, j))
        return results
    
    def _resolve_futures(futures):
        with open(_out_file, "a") as out_file:
            for future in futures:
                for result in future.result():
                    words, i, ans, j = result
                    out_file.write(f"{ans}\n")
                    if j == 0:
                        print(f"{_CRED}{i+1}. {words}{_CEND}")
                    print(f"{_CGREEN}{i+1}-{j+1}. {ans}{_CEND}")
                print("\n")

    _N = 10
    with open("censorship.txt", "r") as in_file:
        with ThreadPoolExecutor(max_workers=_N) as executor:
            _futures = []
            for i, line in enumerate(in_file):
                _futures.append(executor.submit(_process_line, line, i))
                if len(_futures) == _N:
                    _resolve_futures(_futures)
                    _futures = []
            _resolve_futures(_futures)
