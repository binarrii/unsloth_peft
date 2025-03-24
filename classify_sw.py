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
    _out_file = "sensitive_words.txt.c"
    if os.path.exists(_out_file):
        os.remove(_out_file)

    prompt_template = textwrap.dedent("""
    - 为 `{}` 这些词汇或短语分类
    - **类别为以下类别之一**:
        - A1-煽动颠覆国家政权、推翻社会主义制度
        - A1-危害国家安全和利益、损害国家形象
        - A1-煽动分裂国家、破坏国家统一和社会稳定
        - A1-宣扬恐怖主义、极端主义
        - A1-宣扬民族仇恨
        - A1-宣扬暴力、淫秽色情
        - A1-传播虚假有害信息
        - A1-其他法律、行政法规禁止的内容
        - A2-民族歧视内容
        - A2-信仰歧视内容
        - A2-国别歧视内容
        - A2-地域歧视内容
        - A2-性别歧视内容
        - A2-年龄歧视内容
        - A2-职业歧视内容
        - A2-健康歧视内容
        - A2-其他方面歧视内容
        - A3-侵犯他人知识产权
        - A3-违反商业道德
        - A3-泄露他人商业秘密
        - A3-利用算法、数据、平台等优势，实施垄断和不正当竞争行为
        - A3-其他商业违法违规行为
        - A4-危害他人身心健康
        - A4-侵害他人肖像权
        - A4-侵害他人名誉权
        - A4-侵害他人荣誉权
        - A4-侵害他人隐私权
        - A4-侵害他人个人信息权益
        - A4-侵犯他人其他合法权益
        - A5-内容不准确，严重不符合科学常识或主流认知
        - A5-内容不可靠，虽然不包含严重错误的内容，但无法对使用者形成帮助
        - Z0-其他
    - 只能选择一个类别, 不能多个 (IMPORTANT!!!)
    - 只需输出一个结果, 不能多个 (IMPORTANT!!!)
    - 只需输出类别名称, 不要输出任何其他内容 (IMPORTANT!!!)
    """)

    def _process_line(line, i):
        words = _unquot(line.strip())
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
            results.append((words, i, ans, j))
        return results
    
    def _resolve_futures(futures):
        with open(_out_file, "a") as out_file:
            for future in futures:
                for result in future.result():
                    words, i, ans, j = result
                    out_file.write(f"{words}@@{ans}\n")
                    if j == 0:
                        print(f"{_CRED}{i+1}. {words}{_CEND}")
                    print(f"{_CGREEN}{i+1}-{j+1}. {ans}{_CEND}")
                print("\n")

    _N = 20
    with open("sensitive_words.txt", "r") as in_file:
        with ThreadPoolExecutor(max_workers=_N) as executor:
            _futures = []
            for i, line in enumerate(in_file):
                _futures.append(executor.submit(_process_line, line, i))
                if len(_futures) == _N:
                    _resolve_futures(_futures)
                    _futures = []
            _resolve_futures(_futures)
