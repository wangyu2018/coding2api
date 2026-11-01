"""通用 OpenAI 兼容上游渠道（``openai_compat``）。

把任意「POST {base_url}/chat/completions + Bearer key」的服务登记为一条
凭证渠道：智谱 GLM Coding Plan、DeepSeek 计费、MiniMax、豆包/火山方舟、
以及本地 devin-2api / one-api / new-api 之类的自建网关，都以同一份凭证
（``{"base_url", "api_key", "models"}``）纳入凭证池统一调度。

与 Kilo / Zen 的差异：

* **有凭证**——`api_key` 随凭证行落库（加密与否由凭证仓库决定），请求时
  注入 `Authorization: Bearer ...`；key 缺失时匿名转发（内网自建网关）。
* **模型列表两种来源**——凭证里给了 `models` 白名单就用它（省一次探测，
  也允许只暴露上游的子集）；留空则回退到 `GET {base_url}/models`。
* **无额度探测**——各家计费接口形状不一，probe_quota 一律 unknown，
  调度按中性处理。
"""

from .client import OpenAICompatClient, OpenAICompatProvider

__all__ = ["OpenAICompatClient", "OpenAICompatProvider"]
