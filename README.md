# Qwen-Image-2.1 全 NPU 推理后端（Orange Pi AiPro 20t / Ascend 310B1）

> 在 20 TOPS、23 GB 统一内存的边缘 NPU 上，让 Qwen-Image-2.1 的 DiT 与 VAE 全部运行在 CANN NPU 上。
> 本仓库提供基于 stable-diffusion.cpp 的后端补丁、构建/运行脚本与全部实测数据；**不包含模型权重、输出图片与提示词**。

**English abstract** — This repository contains a set of patches that make stable-diffusion.cpp
(commit `60423ab`, ggml `d25b121c`) run Qwen-Image-2.1 end-to-end on an Ascend 310B1 NPU
(Orange Pi AiPro 20t): DiT and VAE on the CANN backend, text encoder on CPU. It adds the missing
CANN operators (REPEAT, SCALE+bias, RMS_NORM, FP32 MatMul, CONV_2D), an FP16 retry path for
attention, scheduler/pool fixes, and diagnostic hooks. Measured on-device:
512×1024 / 40 steps in 2.27–2.38 h sampling per image, 512×512 / 20 steps in ≈34 min end-to-end,
with zero NPU kernel faults across 5 consecutive images. Model weights, generated images and
prompts are intentionally excluded.

---

## 1. 结果概览

| 项目 | 结果 | 证据等级 |
| --- | --- | --- |
| 512×1024 / 40 步，cfg 2.4 | 采样 8169.93–8559.49 s（2.27–2.38 h），端到端约 2.4 h/张 | 【实测】连续 5 张 |
| 512×512 / 20 步，cfg 6.0 | 采样 1889.14–1898.89 s（31.5 min），端到端约 34 min | 【实测】2 次 |
| 256×256 / 1 步，cfg 6.0 | 进程内首个请求采样 188.08 s；后续请求 41.40–56.82 s；`SD_KEEP_PARAMS=1` 时 28.08 s | 【实测】 |
| NPU 内核故障 | 上述所有运行 0 次 | 【实测】日志 `fault kernel_name` 计数为 0 |
| DiT + VAE 所在设备 | 均为 `CANN0`；文本编码器在 CPU | 【实测】`--backend "te=cpu,diffusion=CANN0,vae=CANN0"` |

> 步时不是线性的：512×1024 的稳态约 196 s/步，但第 1 步要 496 s、第 2 步 365 s（首次执行/编译开销），
> 因此 40 步的平均值是 204–214 s/步。【实测】
> 512×512 的稳态是 83.4–84.1 s/步，20 步含首步开销后平均 94.5 s/步。【实测】

---

## 2. 环境

| 组件 | 版本 / 规格 | 证据等级 |
| --- | --- | --- |
| 板卡 | Orange Pi AiPro 20t | 【实测】 |
| NPU | Ascend 310B1，`npu-smi` 显示 310B1，驱动/工具 25.2.0 | 【实测】 |
| 内存 | 统一内存 23,673 MiB（npu-smi 总容量） | 【实测】 |
| CPU | aarch64，3 核 | 【实测】`nproc=3` |
| OS | Ubuntu 22.04.5 LTS，内核 5.10.0+ | 【实测】 |
| CANN | 8.5.1（`/usr/local/Ascend/ascend-toolkit/latest`） | 【查证】按本机安装记录 |
| stable-diffusion.cpp | `60423ab` | 【实测】`git log -1` |
| ggml 子模块 | `d25b121c` | 【实测】`git -C ggml log -1` |
| 构建类型 | Release，`GGML_CANN=ON`，`GGML_NATIVE=ON`，`USE_ACL_GRAPH=OFF` | 【实测】CMakeCache |

### 模型文件（不随仓库分发）

| 用途 | 文件 | 放置位置 |
| --- | --- | --- |
| DiT | `qwen_image_2.1_UC_f16.gguf` | `models/diffusion_models/` |
| VAE | `qwen_image_2.1_vae_f16.safetensors` | `models/vae/` |
| 文本编码器 | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | `models/text_encoders/` |

文本编码器选 Q4_K_M 并在 CPU 上运行，是内存与速度折中的结果：
F16 权重约 16 GB，与 14 GB 的 F16 DiT 无法同时放进 23 GB 统一内存。【推断，基于文件大小与实测可用内存】

---

## 3. 后端改动

补丁分两部分：

* `patches/0001-sd-cpp-310b-backend.patch` — 主仓库，6 个文件，+108/−2。
* `patches/0002-ggml-cann-310b-backend.patch` — ggml 子模块，7 个文件，+840/−26。

`files/` 下是打完补丁后的完整文件副本，方便直接对照；`files/ggml/src/ggml-cann/aclnnop_shim/`
是一个**未被上游跟踪的新增头文件**，补丁无法表达新增文件，构建前必须手动复制（见 §4）。

### 3.1 算子补齐（`ggml-cann/aclnn_ops.cpp`）

310B1 只提供 FP16 的 MatMulV2 / Conv2D_5HD 内核，且没有量化的 matmul 算子。以下算子按 CANN
现有 aclnn 原语组合实现：

| 算子 | 实现方式 | 原因 |
| --- | --- | --- |
| `REPEAT` | 按增长维逐步 `Cat`；多段时用两个 `nbytes(dst)` 的 ping-pong 缓冲；`nGrow == 0` 时直接 `InplaceCopy` | 310B 无原生 repeat，且一次性展开会超出单次拷贝上限 |
| `SCALE`（带 bias） | 先 `Muls`，再用 `InplaceAdds` 加 bias | bias 走 `extra_ops`，原实现忽略 |
| `RMS_NORM` | Cast/Muls → Mul → Mean → InplaceAdds(eps) → Rsqrt → Mul → Cast | 310B 无 RMSNorm 融合算子 |
| FP32 `MUL_MAT` | 输入/权重 Cast 到 FP16 暂存，乘完再 Cast 回 FP32 | 310B 只有 FP16 MatMulV2 |
| `CONV_2D` | FP16 `Conv2D_5HD` + 暂存缓冲，含类型/连续性/退化 shape 检查 | VAE 解码需要，原后端未派发 |
| FlashAttention | 不启用；`supports_op` 对 `FLASH_ATTN_EXT` 返回 false | 310B1 无 `FlashAttentionScore` 内核 |

### 3.2 注意力回退（`src/core/ggml_extend.cpp`）

如果调度器把 `k @ q` 分到了不支持的设备上（310B 对某些类型组合返回 false），就在同一后端内
把 `k`、`q` 转成 F16 后重试一次，避免整个注意力节点被拉回 CPU：

```cpp
auto kq = ggml_mul_mat(ctx, k, q);
if (!ggml_backend_supports_op(backend, kq)) {
    kq = ggml_mul_mat(ctx, ggml_cast(ctx, k, GGML_TYPE_F16), ggml_cast(ctx, q, GGML_TYPE_F16));
}
```

`kqv = kq @ v` 同样处理。

### 3.3 调度与内存

| 改动 | 位置 | 作用 |
| --- | --- | --- |
| 调度器哈希表扩容 | `ggml/src/ggml-backend.cpp` | 原实现按 `graph_size` 分配哈希表，未计入 leaf；NPU 分段后 leaf 数可超容量。改为 `graph_size * 8 + 32768` |
| 设备内存安全余量 | `src/model_manager.cpp` | 新环境变量 `SD_SAFETY_MARGIN_MB`（0–4096，默认 512），补偿静态估算未覆盖的运行时分配 |
| 缓冲池不回收 | `ggml-cann/ggml-cann.cpp` | 310B 上 `GGML_CANN_ACL_GRAPH` 默认 on 时，池从进程启动就不再 `aclrtFree`；`GGML_CANN_DISABLE_BUF_POOL_CLEAN` 可双向覆盖 |
| 主机内存对齐 | `ggml-cann/ggml-cann.cpp` | `aclrtMallocHost` 返回 256 字节对齐切片，原指针藏在切片前一个字 |
| 参数常驻 | `src/core/ggml_runner.cpp` | `SD_KEEP_PARAMS=1` 时请求之间不 evict DiT/VAE 参数 |
| BF16→F16 加载 | `src/model_loader.cpp` | 允许 `--tensor-type-rules` 把 BF16 存储的张量按 F16 请求加载 |

### 3.4 诊断钩子

| 钩子 | 作用 |
| --- | --- |
| `SD_DUMP_DIR` | 把 `noise` / `latent` 以 `int64 ndim | shape[ndim] | float32` 格式落盘，用于逐位回归 |
| `GGML_CANN_SYNC_DEBUG` | 逐节点打印并同步，定位出错的算子 |
| `GGML_PROBE_NMSE` | 覆盖 test-backend-ops 的 NMSE 阈值，便于在低精度后端上跑测试 |

---

## 4. 构建

```bash
# 1. 取基础代码
git clone https://github.com/leejet/stable-diffusion.cpp sd-cpp
cd sd-cpp
git checkout 60423ab
git submodule update --init --recursive
git -C ggml checkout d25b121c

# 2. 打补丁
git apply /path/to/patches/0001-sd-cpp-310b-backend.patch
git -C ggml apply /path/to/patches/0002-ggml-cann-310b-backend.patch

# 3. 补上补丁无法表达的新增头文件
cp -r /path/to/files/ggml/src/ggml-cann/aclnnop_shim ggml/src/ggml-cann/

# 4. 构建
. /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build-cann \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CANN=ON \
      -DGGML_NATIVE=ON \
      -DUSE_ACL_GRAPH=OFF
cmake --build build-cann -j4 --target sd-cli sd-server test-backend-ops
```

产物：`build-cann/bin/{sd-cli,sd-server,test-backend-ops}`。

说明：

* `CANN_INSTALL_DIR` 未显式传入时，`ggml-cann/CMakeLists.txt` 会从 `ASCEND_TOOLKIT_HOME` 推导，
  所以必须先 `source set_env.sh`。
* `USE_ACL_GRAPH` 必须为 `OFF`。上游在 310P 上才允许打开；本板尝试打开时
  `aclmdlRICaptureBegin()` 直接失败（`EH9999 ... Get Allocation Granularity failed`，rc=134），
  因此本后端走 eager 执行。
* 链接使用 `libascendcl.so` / `libnnopbase.so` / `libopapi.so` / `libacl_op_compiler.so` 的完整路径，
  避免板端 `ldconfig` 版本不一致。

---

## 5. 运行

### 5.1 单次生成（sd-cli）

```bash
cd /path/to/qwen-image-2.1
. /usr/local/Ascend/ascend-toolkit/set_env.sh
SD_DUMP_DIR=/tmp/latent \
build-cann/bin/sd-cli \
  --diffusion-model models/diffusion_models/qwen_image_2.1_UC_f16.gguf \
  --vae            models/vae/qwen_image_2.1_vae_f16.safetensors \
  --llm            models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  -p "a lovely cat holding a sign says qwen2.1.cpp" \
  --steps 20 --cfg-scale 6.0 --sampling-method euler --threads 3 \
  --vae-conv-direct --fa -s 42 -v -W 512 -H 512 \
  --backend        "te=cpu,diffusion=CANN0,vae=CANN0" \
  --params-backend "diffusion=CANN0,te=cpu,vae=CANN0" \
  -o output/cat.png
```

要点：

* `--backend` 与 `--params-backend` 必须**同时**指定；只指定其中一个，权重或计算图会退回 CPU。
* `--threads 3` 与 3 个 CPU 核对齐；再大不会更快（文本编码器是 CPU 侧瓶颈）。
* `--fa` 在 310B 上不会启用融合注意力（见 §3.1），保留它只是为了与其它设备共用命令行。

### 5.2 常驻服务（sd-server）

```bash
SD_KEEP_PARAMS=1 build-cann/bin/sd-server -l 127.0.0.1 --listen-port 1241 \
  --diffusion-model models/diffusion_models/qwen_image_2.1_UC_f16.gguf \
  --vae            models/vae/qwen_image_2.1_vae_f16.safetensors \
  --llm            models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --threads 3 --vae-conv-direct --fa --sampling-method euler --cfg-scale 6.0 \
  --backend        "te=cpu,diffusion=CANN0,vae=CANN0" \
  --params-backend "diffusion=CANN0,te=cpu,vae=CANN0"

curl -s http://127.0.0.1:1241/sdapi/v1/txt2img -H 'Content-Type: application/json' -d '{
  "prompt": "a lovely cat holding a sign says qwen2.1.cpp",
  "width": 512, "height": 512, "steps": 20, "cfg_scale": 6.0,
  "seed": 42, "sampler_name": "euler", "batch_size": 1, "n_iter": 1
}'
```

`SD_KEEP_PARAMS=1` 的收益与代价【实测】：

| 指标 | 不设置 | 设置 |
| --- | --- | --- |
| 同进程第 3 个 1 步 256×256 请求 | 44.58 s | **28.08 s**（−37%） |
| 图像 md5 | — | 不变 |
| 常驻内存 | — | 约 2.6 GB |

### 5.3 环境变量

| 变量 | 取值 | 默认 | 说明 |
| --- | --- | --- | --- |
| `SD_KEEP_PARAMS` | 任意非空 | 未设置 | 请求间保留 DiT/VAE 参数 |
| `SD_SAFETY_MARGIN_MB` | 0–4096 | 512 | 设备内存安全余量 |
| `SD_DUMP_DIR` | 目录 | 未设置 | 导出 `noise.bin` / `latent.bin` |
| `GGML_CANN_ACL_GRAPH` | on/off | on（310B） | 决定缓冲池是否从启动期就不回收 |
| `GGML_CANN_DISABLE_BUF_POOL_CLEAN` | 0/1 | 未设置 | 直接覆盖缓冲池回收策略 |
| `GGML_CANN_SYNC_DEBUG` | 任意非空 | 未设置 | 逐节点同步与打印 |
| `GGML_PROBE_NMSE` | 浮点 | 未设置 | 覆盖 backend-ops 的 NMSE 阈值 |

---

## 6. 回归基线

1 步 256×256、seed 42、cfg 6.0、`--fa`、DiT/VAE 在 CANN0 时，导出的 latent 应为：

```
md5(latent.bin) = b747de1e3b97409172bbcc2d8fba39ad
```

命令见 `scripts/smoke-1step.sh`。重建二进制后先跑这一条，再跑长任务。【实测】
注意：`SD_DUMP_DIR` 写入的是 float32 原始张量，文件格式为
`int64 ndim | int64 shape[ndim] | float32 payload`。

---

## 7. 已知限制

| 限制 | 影响 | 现状 |
| --- | --- | --- |
| ACL 图捕获不可用（`EH9999`） | 每步都有 eager 派发开销 | `USE_ACL_GRAPH=OFF` |
| 310B1 无 `FlashAttentionScore` | 注意力是分解实现，慢 | `supports_op` 返回 false，自动回退 |
| 无量化 matmul | 权重必须是 F16/BF16 | Q4/Q8 只用于 CPU 侧的文本编码器 |
| 文本编码器在 CPU | 短提示约 84 s，长提示约 236 s/次 | 可配合 server 复用，避免重复编码 |
| 单进程单任务 | 并发会换页锁死而非变慢 | 运行脚本用 `pgrep` 互斥 |

> 并发一条补充：板上同时跑两个 `sd-cli` 曾出现 26 分钟 0 步的换页抖动，故所有脚本都先检查
> `pgrep -x sd-cli` / `pgrep -x sd-server`。【实测】

---

## 8. 仓库结构

```
.
├── README.md
├── patches/
│   ├── 0001-sd-cpp-310b-backend.patch      # 主仓库补丁
│   └── 0002-ggml-cann-310b-backend.patch   # ggml 子模块补丁
├── files/                                  # 打完补丁后的完整文件（便于对照）
│   ├── sd-cpp/src/...
│   └── ggml/...
├── scripts/
│   ├── build-cann.sh                       # 构建
│   ├── run-sd-cli.sh                       # 单次生成
│   ├── run-sd-server.sh                    # 常驻服务
│   └── smoke-1step.sh                      # 1 步回归
└── licenses/
    ├── LICENSE-stable-diffusion.cpp        # MIT
    └── LICENSE-ggml                        # MIT
```

**不包含**：模型权重、生成图片、提示词、板端账号与网络信息。

---

## 9. 许可证与致谢

* 补丁基于 [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)（MIT）与
  [ggml](https://github.com/ggerganov/ggml)（MIT），许可证文本见 `licenses/`。
* `aclnnop_shim/aclnnop/aclnn_fused_infer_attention_score_v2.h` 来自 CANN Open Software，
  按其头文件内声明的 CANN Open Software License Agreement Version 1.0 使用。
* 模型权重请从官方渠道获取，并遵守其各自的许可；本仓库不提供任何权重。

---

*本文所有性能数字均标注了证据等级：*【实测】*为板上日志直接读数，*【查证】*为安装记录，
【估算】*为按已知步时推算，*【推断】*为基于文件大小/内存的定性判断。*
