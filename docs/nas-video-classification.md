# 绿联 NAS 视频分类：文件名、目录与三层匹配

本分支新增 `classify` 过滤器。只读取视频文件名和所属目录，不转写音频、不读取画面或字幕。
无需模型即可使用关键词规则；可选接入第三方 OpenAI 兼容接口（OpenAI-compatible API），用于语义匹配和大语言模型（LLM）复核。

## 快速使用

从本仓库安装，不要安装上游 PyPI 包（上游没有本功能）：

```bash
python -m pip install .
organize check examples/nas/video-classify.yaml
organize sim examples/nas/video-classify.yaml -F jsonl
organize run examples/nas/video-classify.yaml -F jsonl
```

先修改示例里的扫描路径和目标目录。输入目录与输出目录应分离，不要把输出目录放在扫描目录里。
`sim` 只模拟移动/写入，不修改视频文件；如果启用了远端模型，模拟阶段也会调用模型并产生服务商费用。

## 文件名与目录上下文

例如 `创作者运营/达人孵化课程/01.mp4`：`01` 是无主题标题，分类使用父目录。
`音乐人/第三课_激励.mp4`：文件名提供动作“激励”，目录补充对象“音乐人”。
`创作者运营/向量数据库.mp4`：标题与目录分别命中不同明确类别，进入复核，不按总分静默归档。

- `depth: 3`：最多读取三层祖先，包括被忽略的层，最近父目录在前。
- `include_root: false`：默认不使用扫描根目录名称，不读取扫描根目录之外的目录。
- `ignore_dirs`：忽略无主题目录，按完整目录名称匹配，大小写和全半角不敏感。
- 根目录外的符号链接文件不分类。扫描路径中的 `..` 会先规范化，再校验边界。
- `aliases`：如 `creator: 创作者`，保留原文并补充标准词。原文用于复核证据。
- 纯数字、常见相机前缀与通用词组成的标题不提供主题；更复杂的编号命名需要自行调整预处理逻辑。

目录上下文用于每个文件的独立判断，**不会把第一个视频的类别缓存成整个目录的类别**。
保留完整原相对目录作为 `classify.relative_parent`，避免课程与合集被拆散或不同课程的 `01.mp4` 混在一起。

## 类别与关键词

```yaml
rules:
  - locations: /data/incoming
    subfolders: true
    filters:
      - extension: [mp4, mkv, mov]
      - classify:
          aliases:
            KOL: 达人
          categories:
            - id: creator_operations
              folder: 运营/创作者运营
              description: 创作者和音乐人的招募、孵化、激励及留存
              tags: [运营, 创作者]
              strong: [创作者运营, 达人孵化]
              any: [音乐人, 激励]
              all: [达人, 招募]
              exclude: [员工招聘]
              examples: [如何让音乐人持续投稿]
    actions:
      - move:
          dest: /data/library/{classify.folder}/{classify.relative_parent}/
          on_conflict: skip
```

| 字段 | 逻辑 |
| --- | --- |
| `id` | 唯一分类标识，大模型只能返回候选标识 |
| `folder` | 相对目标目录，禁止绝对路径和 `..` |
| `description`、`examples` | 给语义模型和大模型的类别定义与正例 |
| `tags` | 接受该类别后返回的固定标签，不决定移动目录 |
| `strong` | 每个命中短语加 8 分 |
| `any` | 每个命中词加 3 分 |
| `all` | 所有词都出现时加 6 分，可跨文件名与目录组合 |
| `exclude` | 任何排除词命中就取消该类别，后续模型也不能选回 |

默认 `keyword_threshold: 6`、`keyword_margin: 2`。最高分同时达到阈值并领先第二名才接受；平局进入后续层。
每个词每个类别只计一次，不因同一词在多个目录反复出现而加分。英文词采用字母/数字边界匹配，`RAG` 不会命中 `dragon`；中文按子串匹配。
文件名与目录单独得到不同明确类别时判定冲突。不同祖先目录的多主题不做强制继承；总分接近时保留待确认。
分数代表规则强度，不是概率。避免在 `strong` 和 `any` 中重复定义相同词导致权重叠加。

## 第三方远端模型

在 `classify` 中添加：

```yml
model_api:
  base_url: https://YOUR-PROVIDER/v1
  api_key_env: NAS_MODEL_API_KEY
  llm_model: YOUR-CHAT-MODEL
  # 如果服务商提供兼容的嵌入接口，再启用此行：
  # embedding_model: YOUR-EMBEDDING-MODEL
  json_mode: true
  timeout: 60
  max_tokens: 1024
  top_k: 3
  threshold: 0.75
  margin: 0.10
```

`base_url` 是接口前缀，不是完整 `/chat/completions` 地址；保留服务商需要的 `/v1` 或其他前缀。
密钥只从 `api_key_env` 指定的环境变量读取，不写进 YAML、代码或 Git。
使用服务商给出的 HTTPS 地址。请求不跟随重定向，避免认证信息传到其他地址。

接口格式：

- 对话：`POST <base_url>/chat/completions`，请求包含 `model`、`messages`、`temperature`、`max_tokens`、`stream: false`；读取 `choices[0].message.content`。
- 向量：`POST <base_url>/embeddings`，请求包含 `model`、`input`；读取 `data[0].embedding`。
- 认证：`Authorization: Bearer <环境变量中的密钥>`。
- `json_mode: true` 增加 `response_format: {type: json_object}`；不支持此参数的服务商改为 `false`，返回内容仍必须是有效 JSON，不接受 Markdown 代码块包裹。

远端输入包括文件名、有限目录名称、原相对父目录、类别定义，不上传视频文件或绝对 NAS 路径。

### 三层执行顺序

1. 关键词可靠命中：直接结束，不调用模型。
2. 若配置了 `embedding_model`：比较上下文与各类别“标识＋定义＋正例＋关键词”的余弦相似度（cosine similarity）。最高分达阈值且差值达 `margin` 才接受；明确标题/目录冲突不由向量直接裁决。
3. 若仍不确定且配置了 `llm_model`：提交排名前 `top_k` 的类别。冲突时额外保留标题和目录类别，不受 `top_k` 限制。
4. 大模型只能选候选类别或返回待确认，必须提供文件名/目录原文中的非空证据。拒绝未知类别、虚构证据或 `needs_review: true`。
5. 超时、认证失败、畸形响应不导致移动。嵌入接口不可用时仍可尝试对话层；对话层不可用则待确认。

返回示例：

```json
{"category_id":"creator_operations","needs_review":false,"evidence":["达人孵化"]}
```

语义阈值需要用你人工标注的真实文件名校准，示例值不保证分类准确率。
大模型只判断名称，不推测视频真实内容；原文证据校验不是正确率保证，首次务必检查模拟结果。
向量缓存在单个过滤器实例内最多保留 512 条，重复上下文/类别描述可复用；不跨运行持久化。

## 预览、待确认和日志

`match` 默认是 `accepted`，只让已确定文件进入动作链：

- `accepted`：分类后移动。
- `review`：只处理待确认结果，适合配合 `echo` 或 `write` 生成复核清单。
- `all`：输出全部分类报告，适合只配 `echo` 或 `write`。**不要将 `all`/`review` 直接接分类目录移动动作**，待确认结果的 `folder` 为 null。

所有分类结果都会打印为一条 `classify` 消息，包括未匹配和模型失败。
使用 `-F jsonl` 得到包含分类消息、移动消息和最终报告的结构化日志；分类消息的 `msg` 是一个 JSON 字符串，需要再解析一次。

可用变量：`classify.status`、`category_id`、`folder`、`stage`、`reason`、`evidence`、`tags`、`candidates`、`context`、`relative_parent`、`source`、`json`。
语义层运行后还会提供 `semantic_candidates`；没有运行时此字段不存在。
规则名称、阈值和类别配置以你保留的配置文件为准。

示例采用 `on_conflict: skip`，不覆盖同名文件。分类过滤器不会移动文件，实际移动沿用 Organize 原有动作。
示例写入的 `classifications.jsonl` 是分类决策记录，**不等同于成功移动清单**：遇到同名冲突时移动会跳过，但分类记录仍写入。
请结合移动消息和最终错误报告核对执行状态。本版本不提供自动回滚命令。

## 绿联 Docker 部署

在绿联上克隆本分支。确认型号和系统支持容器（Docker），然后进入示例目录：

```bash
cd examples/nas
```

在此目录的本地 `.env` 文件填入真实目录，不要提交密钥：

```dotenv
NAS_INCOMING=/YOUR-NAS-PATH/incoming
NAS_LIBRARY=/YOUR-NAS-PATH/library
NAS_LOGS=/YOUR-NAS-PATH/organize-logs
NAS_MODEL_API_KEY=
```

`NAS_MODEL_API_KEY` 只在启用模型时需要。三处目录应提前建立，服务运行用户需有读写权限。
不要把库目录或日志目录放进扫描目录，避免下一次扫描重复处理。

```bash
# 构建当前分支并预览；默认命令就是 sim。
docker compose --profile organize run --rm --build organize

# 人工核对预览后正式执行；终端日志包含待确认与移动消息。
docker compose --profile organize run --rm organize run /config/video-classify.yaml -F jsonl
```

该示例是按需执行的批处理，不是网页应用。可由绿联定时任务调用正式执行命令。
容器权限取决于你设备的目录权限设置；不确定时先只运行模拟并核对目录映射。

## 验证边界

自动化测试覆盖目录继承、跨来源组合、冲突、排除条件、扫描边界、无主题名称、模拟移动、同名冲突与兼容接口的模拟响应。
没有实际连接用户的 NAS，也没有使用真实第三方密钥做远端测试。服务商的模型名、参数支持和分类效果需要配置后实测。
