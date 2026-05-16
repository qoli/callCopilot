# callCopilot

Standalone launcher for GitHub Copilot CLI BYOK providers.

The launcher prefers macOS system Python at `/usr/bin/python3` and uses only the Python standard library. Provider proxies use Node.js because they sit on the HTTP streaming boundary.

## Usage

```bash
bin/callCopilot ds -- -s -p "Reply OK only."
bin/callCopilot dsf -- -s -p "Reply OK only."
bin/callCopilot nds -- -s -p "Reply OK only."
bin/callCopilot Qwen3.6-35B-A3B-bf16 -- -s -p "Reply OK only."
```

Aliases:

- `ds`: DeepSeek official Anthropic API, `deepseek-v4-pro`, with thinking block preservation.
- `dsf`: DeepSeek official Anthropic API, `deepseek-v4-flash`, with thinking block preservation.
- `nds`: NVIDIA hosted DeepSeek OpenAI API, `deepseek-ai/deepseek-v4-pro`, with `chat_template_kwargs.thinking=false`.
- Any other model id: local oMLX OpenAI-compatible backend, with response cleanup for nullable OpenAI-compatible fields.

## Config

Create `.env` from `examples/.env.example`.

Required for online providers:

```env
DEEPSEEK_API_KEY=...
NVIDIA_DEEPSEEK_API_KEY=...
```

Local oMLX defaults:

```env
CALLCOPILOT_OMLX_SERVER_ROOT=http://127.0.0.1:8001
CALLCOPILOT_OMLX_API_KEY=change-me
```

## Runtime Dependencies

- `/usr/bin/python3`
- `copilot`
- `node`
- `npm` only when `ds` or `dsf` first installs `opencode-deepseek-thinking-fix`

`ds` and `dsf` install `opencode-deepseek-thinking-fix` into `.runtime/deepseek-thinking-fix/`.
