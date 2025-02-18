#!/bin/bash

curl -sN http://localhost:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '
        {
          "model": "Qwen2.5-14B-Screenplay-ft",
          "messages": [
            {
              "role": "system",
              "content": "你是一位优秀的中文编剧，非常擅长写剧本，包括多幕剧和独幕剧"
            },
            {
              "role": "user",
	      "content": "写一个完整的中文剧本, 包括剧名、人物和角色设定、剧情发生的时间地点、每一幕的具体情节和人物对话台词, 分章节共写5幕, 不要输出与剧本无关的其他内容"
            }
          ],
	  "temperature": 0.7,
	  "frequency_penalty": 1.0,
	  "max_completion_tokens": 3200,
          "stream": true
        }     
        ' \
     | jq --unbuffered -Rrj 'ltrimstr("data: ") | fromjson? | .choices[0].delta.content | select( . != null )' && printf '\n\n'
