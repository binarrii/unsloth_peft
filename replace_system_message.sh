#!/bin/bash

sed -i 's/You are Qwen, created by Alibaba Cloud. You are a helpful assistant./You are an excellent Chinese screenwriter, very skilled at writing screenplays./g' models/Qwen2.5-14B-Screenplay-ft/tokenizer_config.json
