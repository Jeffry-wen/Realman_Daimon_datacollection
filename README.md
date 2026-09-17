# RealMan X5 多模态对齐数据采集工程

本工程用于采集可直接用于机器人学习训练的 LeRobot v3 数据集。系统面向 RealMan RM75b 双臂平台，把机械臂状态与动作、UGripper 夹爪、腕部 RGB、X5 触觉和六维力数据按统一时间轴对齐，并按 episode 事务化保存。

默认工作方式是 `drag` 拖动示教：操作者直接拖动真实从臂，主臂只提供夹爪开合信号。采集过程中如果必需传感器缺帧、时间戳过旧、编码器阻塞或控制循环持续超时，当前 episode 会报错退出，而不是悄悄复用旧数据。

## 1. 默认配置概览

| 项目 | 默认值 |
| --- | --- |
| 采集对象 | 左、右双臂 |
| 控制模式 | `drag` 拖动示教 |
| 采样频率 | 30 FPS |
| 腕部 RGB | 1920×1080@30 输入，去畸变后保存 |
| X5 触觉 | 四路，`depth_deformation`，每路 3×96×128 |
| 六维力 | 左右臂各一组 Fx/Fy/Fz/Mx/My/Mz |
| 夹爪 | UGripper，主臂夹爪信号控制从臂夹爪 |
| 编码 | FFmpeg `h264_nvenc` 流式编码 |
| Episode 操作 | 网页或三键脚踏板开始、保存、丢弃 |
| 软件急停 | 双踏板组合默认关闭；使用物理急停 |
| 数据格式 | LeRobot v3 |
| 自动回位 | 固定回合前回位默认关闭；当前机器单臂动态回位已开启 |

当前机器的本地配置是 `config/lab_5090.env`。这个文件被 Git 忽略，不会覆盖其他机器的配置；可提交的模板是 `config/lab_5090.env.example`。

## 2. 安全须知

在给机械臂上电或正式采集之前，必须满足以下条件：

1. 操作者能够立即触及机械臂的物理急停按钮。
2. 工作空间内没有无关人员、线缆、工具或其他障碍物。
3. 第一次运行、修改网络配置或修改回位姿态时，不安装危险末端工具和易碎负载。
4. 先执行 `scripts/run_capture.sh --check`，通过后再启动真实硬件。
5. 确认 `EPISODE_RIGHT_ARM_RESET_ENABLED=false`；若启用动态回位，还要确认工作空间允许右从臂从任务终点低速返回本回合起点。

当前机器已关闭双踏板软件急停，使开始、保存和丢弃命令在踩下后立即入队。请始终确保机械臂的物理急停按钮触手可及。可选的软件急停依赖 Linux、USB 输入、Python 进程、网络和厂商 SDK，**不是安全等级急停，不能替代机械臂的物理急停回路**。

更完整的安全说明见 [docs/SAFETY.md](docs/SAFETY.md)。

## 3. 系统组成

### 3.1 采集链路

```text
scripts/run_capture.sh
  ├─ 加载 config/lab_5090.env
  ├─ 执行 scripts/preflight_capture.py
  ├─ 停止旧采集进程（正式启动时）
  └─ scripts/capture_realman_x5_force_aligned_app.sh
       └─ scripts/capture_realman_x5_force_aligned_app.py
            ├─ 加载兼容 LeRobot 工程
            ├─ 连接从臂、主臂、夹爪、相机、X5 和力传感器
            ├─ 建立各数据源的时间戳缓存
            ├─ 以统一目标时间选择 observation/action
            └─ 保存 LeRobot v3 episode
```

本仓库只包含采集层。机械臂、主臂、UGripper、X5、数据集写入器和腕部相机驱动来自单独的兼容 LeRobot checkout，由 `LEROBOT_ROOT` 指定。详细依赖关系见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

### 3.2 主要文件

| 路径 | 用途 |
| --- | --- |
| `config/lab_5090.env.example` | 可提交的机器配置模板 |
| `config/lab_5090.env` | 当前机器实际配置，Git 忽略 |
| `scripts/run_capture.sh` | 双臂主启动入口、预检和旧进程清理 |
| `scripts/run_right_arm_capture.sh` | 右臂单臂采集入口 |
| `scripts/capture_realman_x5_force_aligned_app.py` | 对齐采集、拖动示教、脚踏板和 episode 主逻辑 |
| `scripts/capture_realman_x5_force_app.py` | 对兼容 LeRobot 采集程序的配置和运行时适配 |
| `scripts/preflight_capture.py` | 不打开硬件的配置预检 |
| `scripts/listen_foot_pedal.py` | 查找脚踏板并识别按键码 |
| `scripts/stop_capture_app.sh` | 停止当前采集及其子进程 |
| `scripts/verify_bundle.sh` | Shell、Python 和硬件无关单元测试 |
| `capture_support/` | 触觉帧桥接支持代码 |
| `tests/` | 拖动、回位、主臂缺失回退和双踏板急停测试 |

## 4. 当前硬件和网络约定

以下地址来自兼容 LeRobot 驱动的机器配置：

| 设备 | 左侧 | 右侧 |
| --- | --- | --- |
| RealMan 从臂 | `192.168.1.201:8080` | `192.168.1.200:8080` |
| X5/夹爪控制盒 | `192.168.1.10` | `192.168.1.11` |
| UGripper 服务 | `192.168.1.10:55551` | `192.168.1.11:55551` |
| 腕部相机 UDP | `56010` | `56020` |
| 主臂串口（驱动逻辑槽位） | FTDI `DU0E1ZZP` | FTDI `DU0E2O5H` |

X5 Flux 数据发往采集机 `192.168.1.100`，四路接收端口为：

| 触觉位置 | UDP 端口 |
| --- | --- |
| 左夹爪左侧 | `61000` |
| 左夹爪右侧 | `61001` |
| 右夹爪左侧 | `61002` |
| 右夹爪右侧 | `61003` |

稳定的主臂设备路径是：

```text
/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0E1ZZP-if00-port0
/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0E2O5H-if00-port0
```

应使用 `/dev/serial/by-id/`，不要在配置中依赖可能随重启变化的 `/dev/ttyUSB0`、`/dev/ttyUSB1`。

本机已经通过实时开合测试确认：物理右主臂实际接在驱动的左逻辑槽位
`DU0E1ZZP`。因此右臂单臂采集需要开启交叉映射；这里的“逻辑槽位”不代表
机械安装位置。

## 5. 软件环境

推荐环境：

- Linux；
- Conda 环境 `lerobot51`；
- Python、NumPy、OpenCV 和兼容 LeRobot 依赖；
- FFmpeg，并支持当前配置要求的 `h264_nvenc`；
- NVIDIA 驱动可正常使用 NVENC；
- 当前用户能够读取主臂串口和 `/dev/input` 脚踏板设备；
- 足够的磁盘空间用于 RGB 和触觉视频。

兼容 LeRobot 根目录至少应包含：

```text
tools/bi_x5_capture_app.py
src/lerobot/robots/bi_realman_ugripper_notac_new/
src/lerobot/robots/realman_ugripper_notac_new/
src/lerobot/teleoperators/bi_realman_rm75b_leader/
src/lerobot/teleoperators/realman_rm75b_leader/
```

## 6. 首次配置

进入工程：

```bash
cd /home/kunpeng/lerobot-git/realman-x5-aligned-capture
```

如果本机配置不存在，复制模板：

```bash
cp config/lab_5090.env.example config/lab_5090.env
```

至少修改 LeRobot 路径：

```bash
LEROBOT_ROOT=/home/kunpeng/lerobot-git/lerobot-0.5.1-main-20260520-sunday
CONDA_ENV=lerobot51
```

设置数据输出目录。当前机器使用：

```bash
DATASET_BASE_DIR=/home/kunpeng/lerobot-git/kd-tacmae/dataset
```

模板默认把数据写到本工程的 `dataset/`，该目录同样被 Git 忽略。

### 6.1 脚踏板识别

列出输入设备：

```bash
conda run --no-capture-output -n lerobot51 \
  python scripts/listen_foot_pedal.py --list
```

查看三个踏板产生的按键。把设备路径替换为实际的 `/dev/input/by-id/...-event-kbd`：

```bash
conda run --no-capture-output -n lerobot51 \
  python scripts/listen_foot_pedal.py \
  --device /dev/input/by-id/usb-PCsensor_FS20Pro-event-kbd \
  --presses 3
```

然后把实际路径和按键写入 `config/lab_5090.env`：

```bash
FOOT_PEDAL_CONTROL=true
FOOT_PEDAL_DEVICE=/dev/input/by-id/usb-PCsensor_FS20Pro-event-kbd
FOOT_PEDAL_START_KEY=KEY_A
FOOT_PEDAL_FINISH_KEY=KEY_B
FOOT_PEDAL_DISCARD_KEY=KEY_C
FOOT_PEDAL_TWO_PEDAL_STOP_ENABLED=false
FOOT_PEDAL_TWO_PEDAL_WINDOW_S=0.0
```

若提示没有权限，检查：

```bash
ls -l /dev/input/by-id/
groups
```

推荐通过 `input` 组和 udev 规则长期授权，修改用户组后重新登录。不要把每次执行 `chmod 666` 当作固定部署方案。

### 6.2 基础设备检查

```bash
# 主臂串口
ls -l /dev/serial/by-id/*FTDI*

# 从臂和控制盒网络
ping -c 1 192.168.1.200
ping -c 1 192.168.1.201
ping -c 1 192.168.1.10
ping -c 1 192.168.1.11

# 编码器
ffmpeg -hide_banner -encoders | grep h264_nvenc

# 磁盘空间
df -h /home/kunpeng/lerobot-git/kd-tacmae/dataset
```

`ping` 成功只能说明主机可达，不代表 SDK、相机、夹爪或传感器一定能够连接。

## 7. 关键配置说明

通常只需修改 `config/lab_5090.env`。不要直接在很长的启动命令中重复覆盖几十个变量，否则难以复现每次采集使用的配置。

### 7.1 任务和数据集

| 配置项 | 说明 |
| --- | --- |
| `TASK_INDEX` | `both`、`right` 或 `left`；默认 `both` |
| `TASK_BOTH` | 双臂训练任务的自然语言描述 |
| `TASK_RIGHT` | 右臂训练任务描述 |
| `TASK_LEFT` | 左臂训练任务描述 |
| `FPS` | 数据集帧率，默认 30 |
| `NUM_EPISODES` | 本次最多保存多少个 episode |
| `EPISODE_TIME_S` | 单个 episode 最大时长，达到后自动保存 |
| `RUN_NAME_PREFIX` | 双臂数据集名称前缀 |
| `DATASET_BASE_DIR` | 数据集父目录 |

任务描述会写入每一帧。应使用具体、稳定、能描述成功条件的训练指令，例如：

```bash
TASK_BOTH="Pick up the bottle with both arms and place it in the tray"
```

不要在同一个数据集中途改变任务语义。如果任务发生变化，应停止当前程序，用新的数据集名称重新启动。

### 7.2 控制模式

| 模式 | 从臂关节 | 从臂夹爪 | 数据中的关节 action |
| --- | --- | --- | --- |
| `drag` | 人工拖动，不跟随主臂关节 | 跟随主臂夹爪 | 从臂自身关节状态 |
| `leader` | 只有显式启用 `SEND_ACTION_ENABLED=true` 才会发送主臂关节动作 | 跟随主臂夹爪 | 主臂或缺失侧回退值 |
| `program` | 不发送关节动作 | 程序接口提供夹爪目标 | 从臂状态加程序夹爪目标 |

默认配置是：

```bash
CAPTURE_CONTROL_MODE=drag
```

在 `drag` 模式下，主臂的七个关节移动不会驱动真实从臂。程序仍读取主臂，但只使用夹爪信号；从臂关节 action 会被从臂自身的对齐状态覆盖。

### 7.3 传感器和质量保护

| 配置组 | 作用 |
| --- | --- |
| `CONNECT_WRIST_CAMERA`、`WRIST_*` | 腕部 RGB、去畸变和缓存 |
| `CONNECT_X5_TACTILE`、`X5_TACTILE_*` | X5 模态、分辨率、帧率和 UDP 端口 |
| `CONNECT_*_FORCE_SENSOR`、`FORCE_SENSOR_*` | 六维力连接、清零和数据字段 |
| `ALIGNED_*_MAX_AGE_MS` | 各数据源允许的最大时间偏移或年龄 |
| `ALIGNED_ABORT_ON_MISSING` | 缺少必需数据时是否中止 |
| `TACTILE_STALE_*` | 触觉画面重复/停滞检测 |
| `CAPTURE_LOOP_*` | 实时循环持续超时保护 |
| `GRIPPER_MIN_POSITION`、`GRIPPER_MAX_POSITION` | 夹爪位置限幅，范围 0–1000 |
| `GRIPPER_TORQUE_LIMIT` | 夹爪力矩限制，允许范围 10–100 |

预览窗口只影响显示，不改变保存的数据：

```bash
RGB_PREVIEW=true
TACTILE_PREVIEW=true
```

### 7.4 固定回合前回位与动态回到起点

默认关闭：

```bash
EPISODE_RIGHT_ARM_RESET_ENABLED=false
```

开启后，每次 episode 开始前会先退出拖动模式，并用 `rm_movej` 命令右从臂运动到 `EPISODE_RIGHT_ARM_RESET_JOINTS_DEG`，成功后才重新进入拖动模式。左臂不会收到回位目标。

启用前必须清空工作空间、核对七个角度值，并让操作者守在物理急停旁。若当前任意关节与目标相差超过 `EPISODE_RIGHT_ARM_RESET_MAX_START_DELTA_DEG`，程序会拒绝运动。

单臂采集推荐使用动态回位。当前机器配置为：

```bash
EPISODE_RIGHT_ARM_RESET_ENABLED=false
EPISODE_RETURN_TO_START_ENABLED=false
RIGHT_ARM_EPISODE_RETURN_TO_START_ENABLED=true
EPISODE_RETURN_TO_START_SPEED=5
EPISODE_RETURN_TO_START_MAX_DELTA_DEG=60
```

`run_right_arm_capture.sh` 会把 `RIGHT_ARM_EPISODE_RETURN_TO_START_ENABLED` 映射为本次运行的动态回位开关，双臂入口仍保持关闭。硬件连接完成后，右主臂夹爪会立即控制右从臂夹爪，不必先开始录制。左踏板的开始命令被接受时，程序先记录右从臂当时的 7 个关节角和夹爪实际位置，然后才启用拖动并录制。中踏板保存或右踏板丢弃后，程序关闭拖动模式并以 5% 速度让右从臂回到刚才记录的关节位置；夹爪不会自动复位，恢复主臂实时跟随。若任一关节的返回距离超过 60°，程序拒绝运动并进入 `ERROR`。固定回合前回位与动态回位不能同时开启。

## 8. 启动前验证

只检查配置、文件、设备节点、FFmpeg 和 Python 导入，不连接机械臂或相机：

```bash
scripts/run_capture.sh --check
```

正常结束应看到：

```text
[preflight] passed: configuration, local devices, encoder and compatible LeRobot imports are available
[capture-app] DRY_RUN=true, not starting hardware capture.
```

运行代码级验证：

```bash
scripts/verify_bundle.sh
```

该命令执行 Shell 语法检查、Python 编译检查和硬件无关单元测试，不会移动机械臂。

每次修改以下内容后都应重新执行两项验证：

- `config/lab_5090.env`；
- 脚踏板或按键映射；
- 主臂 USB 连接；
- 相机、X5 或网络地址；
- 控制模式、回位参数或夹爪限制；
- Python 和 Shell 代码。

## 9. 双臂采集操作

### 9.1 启动

```bash
cd /home/kunpeng/lerobot-git/realman-x5-aligned-capture
scripts/run_capture.sh
```

正式启动会先运行 preflight，然后尝试停止旧采集进程，再连接硬件。浏览器默认打开：

```text
http://127.0.0.1:8766
```

不要立即开始录制。等待终端和网页状态进入 `READY`，并检查 RGB/触觉预览正常更新。

若需要使用另一份配置：

```bash
scripts/run_capture.sh --config /absolute/path/to/another.env
```

若已经确认没有旧进程且不希望执行停止脚本：

```bash
scripts/run_capture.sh --no-stop
```

### 9.2 Episode 操作

网页提供四个按钮：

| 操作 | 含义 |
| --- | --- |
| 开始录制 | 等待所有数据源新鲜稳定，启用拖动模式并开始新 episode |
| 结束并保存 | 停止拖动，写完队列，保存 episode 和元数据快照 |
| 丢弃当前轮 | 停止拖动并删除当前未保存帧，episode 编号不增加 |
| 停止程序 | 丢弃当前未完成 episode，完成数据集收尾并断开设备 |

脚踏板对应命令由配置中的按键决定：

| 配置项 | 命令 |
| --- | --- |
| `FOOT_PEDAL_START_KEY` | 开始录制 |
| `FOOT_PEDAL_FINISH_KEY` | 结束并保存；动态回位开启时随后低速复位 |
| `FOOT_PEDAL_DISCARD_KEY` | 丢弃当前轮；动态回位开启时随后低速复位 |
| 任意两个已配置踏板同时按下 | 默认不再作为组合急停；分别按当前状态处理 |

当前配置关闭双踏板组合急停，并将组合键等待设为 0，因此单踏板命令会立即入队。若以后显式设置 `FOOT_PEDAL_TWO_PEDAL_STOP_ENABLED=true`，才会恢复组合急停及其等待窗口。

### 9.3 推荐采集流程

1. 检查物理急停、工作空间、线缆和负载。
2. 运行 `scripts/run_capture.sh --check`。
3. 正式运行 `scripts/run_capture.sh`。
4. 等待状态变为 `READY`。
5. 检查腕部 RGB、四路触觉和力传感器没有报错。
6. 踩开始踏板或点击“开始录制”。
7. 在从臂进入拖动模式后完成人工示教。
8. 成功样本踩保存；失败、碰撞或任务未完成时踩丢弃。
9. 等待状态重新回到 `READY` 后再开始下一条。
10. 采集完成后点击“停止程序”或使用停止脚本，并确认数据集完成 finalize。

`finish` 后不要立刻再次踩开始；应等页面从 `SAVING` 回到 `READY`。

### 9.4 运行状态

| 状态 | 说明 |
| --- | --- |
| `INIT` | 程序初始化 |
| `CONNECTING` | 正在连接硬件 |
| `READY` | 可以开始下一条 episode；拖动模式处于关闭状态 |
| `CALIBRATING` | 等待数据缓存、编码器或回位流程完成 |
| `RECORDING` | 正在采集；`drag` 模式下从臂可人工拖动 |
| `SAVING` | 正在保存，暂时不要开始下一条 |
| `ERROR` | 当前运行失败，查看终端中的第一条异常 |
| `STOPPING` / `DONE` | 正在结束或已经结束 |

## 10. 数据内容和目录

数据集目录名称默认由前缀和启动时间组成：

```text
<DATASET_BASE_DIR>/<RUN_NAME_PREFIX>_<YYYYMMDD_HHMMSS>/
```

双臂默认前缀：

```text
kd_tacmae_force_both_20260917_140000
```

典型目录结构：

```text
dataset_root/
  data/                         # 标量状态、动作和索引的 Parquet 数据
  videos/                       # 腕部 RGB 和触觉视频
  meta/                         # info、tasks、episodes 等 LeRobot 元数据
  .aligned_recorder_snapshots/  # 最近保存成功的元数据快照
```

双臂模式通常包含：

- 左右从臂各 7 个关节位置；
- 左右夹爪状态与 action；
- 左右六维力；
- 左右腕部 RGB；
- 左右夹爪各两路 X5 `depth_deformation`；
- `timestamp`、`frame_index`、`episode_index`、`task_index` 和任务文本。

确切 feature、shape、FPS 和 episode 数量以数据集的 `meta/info.json` 为准：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/absolute/path/to/dataset")
info = json.loads((root / "meta" / "info.json").read_text())
print("fps:", info["fps"])
print("episodes:", info["total_episodes"])
for name, spec in info["features"].items():
    print(name, spec.get("dtype"), spec.get("shape"), spec.get("names"))
PY
```

每次成功保存后，程序会校验元数据并在 `.aligned_recorder_snapshots/` 中保留最近的快照。异常退出时，未保存的当前 episode 会被清理；已经保存的 episode 不应被当作临时缓冲删除。

## 11. 数据可视化

使用兼容 LeRobot checkout 的可视化命令。把 `--root` 和 `--repo-id` 替换为启动日志中显示的值：

```bash
conda activate lerobot51
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/kunpeng/lerobot-git/lerobot-0.5.1-main-20260520-sunday

PYTHONPATH=src python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id local/kd_tacmae_force_right_20260908_192141 \
  --root /home/kunpeng/lerobot-git/kd-tacmae/dataset/kd_tacmae_force_right_20260908_192141 \
  --episode-index 0 \
  --video-backend torchcodec \
  --tolerance-s 0.1 \
  --num-workers 0 \
  --batch-size 8 \
  --skip-unreadable-modalities \
  --mode distant \
  --web-port 9090 \
  --grpc-port 9877
```

另一个终端打开 Rerun：

```bash
conda activate lerobot51
rerun rerun+http://127.0.0.1:9877/proxy
```

网页控制台本身也能列出已经保存的 episodes，并预览少量 state/action 数值，但它不能替代完整的数据集可视化和质量检查。

## 12. 停止、异常退出和恢复

优先使用网页“停止程序”，让程序完成 episode 清理、数据集 finalize 和硬件断连。

也可以执行：

```bash
cd /home/kunpeng/lerobot-git/realman-x5-aligned-capture
scripts/stop_capture_app.sh
```

停止脚本会先尝试通知本地 UI，然后查找采集及其子进程，依次发送 `INT`、`TERM`，必要时再发送 `KILL`。不要在同一台机器上有另一个需要保留的采集任务时盲目执行它。

若发生异常：

1. 必要时先按物理急停。
2. 保存终端中最早出现的异常信息；后续错误往往只是连锁反应。
3. 运行 `scripts/stop_capture_app.sh` 清理残留进程。
4. 检查端口、串口、磁盘空间和设备网络。
5. 再运行 `scripts/run_capture.sh --check`。
6. 确认最后一个已保存 episode 可以读取后再继续采集。

不要手工删除正在写入的数据集目录。

## 13. 常见问题

### 13.1 `foot pedal device does not exist` 或权限不足

- 使用 `scripts/listen_foot_pedal.py --list` 重新找设备；
- 优先使用 `/dev/input/by-id/` 路径；
- 检查用户是否属于 `input` 组；
- 拔插脚踏板后再次确认设备链接。

### 13.2 主臂连接失败

```bash
ls -l /dev/serial/by-id/*FTDI*
fuser /dev/ttyUSB0 /dev/ttyUSB1 2>/dev/null
```

确认两个 FTDI 设备都存在，且没有其他进程占用。主臂关节在 `drag` 模式下不会控制从臂，但提供夹爪信号的主臂仍必须能够读取。

### 13.3 从臂连接失败

检查 `192.168.1.200:8080` 和 `192.168.1.201:8080` 是否可达，并确认控制器没有被其他 SDK 进程占用。单臂模式只要求右从臂。

### 13.4 X5 触觉超时、重复或端口占用

```bash
ss -lunp | grep -E ':(61000|61001|61002|61003)\b'
```

确保发送端目标主机是 `192.168.1.100`，四路端口没有被旧采集或其他项目占用。触觉持续重复会触发 stale guard 并中止当前采集，以免写入无效训练数据。

### 13.5 `h264_nvenc` 不可用

```bash
ffmpeg -hide_banner -encoders | grep h264_nvenc
nvidia-smi
```

检查 NVIDIA 驱动与 FFmpeg 构建。若要临时改用其他编码器，必须先确认兼容 LeRobot 的视频写入和读取链路支持该编码器，再修改 `VCODEC`。

### 13.6 磁盘空间不足

腕部 RGB 和四路触觉会快速占用空间。启动前和长时间采集期间都应检查：

```bash
df -h /home/kunpeng/lerobot-git/kd-tacmae/dataset
du -sh /home/kunpeng/lerobot-git/kd-tacmae/dataset/* | sort -h | tail
```

不要在采集运行时移动或压缩当前数据集。

### 13.7 页面一直不进入 `READY`

查看终端日志中最后一个正在连接或等待的数据源。常见原因是腕部相机、X5 Flux、六维力、主臂串口或编码器启动失败。`--check` 不会实际打开这些设备，因此 preflight 通过不等于所有硬件运行时一定正常。

## 14. 开发、测试和版本管理

修改代码后运行：

```bash
scripts/verify_bundle.sh
scripts/run_capture.sh --check
```

CI 和本地验证都不应启动真实机械臂。硬件联调必须由现场操作者在清空工作空间、物理急停可用的条件下单独进行。

以下内容已被 `.gitignore` 排除，不应提交：

- `config/lab_5090.env` 等本机配置；
- 数据集、视频、日志和临时文件；
- Python 缓存和编辑器状态。

本目录可作为独立 Git 仓库维护，但兼容 LeRobot checkout 仍是单独依赖，不包含在本仓库中。

## 15. 单臂采集：右从臂拖动示教

单臂入口是：

```bash
scripts/run_right_arm_capture.sh
```

它会在加载正常机器配置后强制应用以下设置：

```text
TASK_INDEX=right
CAPTURE_CONTROL_MODE=drag
CONNECT_LEFT_FORCE_SENSOR=false
CONNECT_RIGHT_FORCE_SENSOR=true
RUN_NAME_PREFIX=kd_tacmae_force_right
```

### 15.1 单臂模式连接哪些设备

右臂单臂模式只启用：

- 右从臂 `192.168.1.200:8080`；
- 右从臂 UGripper；
- 右腕部 RGB；
- 右夹爪左、右两个 X5 触觉传感器；
- 右臂六维力传感器；
- 当前映射中负责右夹爪信号的一个物理主臂。

左从臂、左腕部 RGB、左侧两个 X5 和左臂六维力不会进入单臂数据集。

### 15.2 主臂与从臂的关系

在默认 `drag` 模式下：

- 主臂七个关节的运动**不会**下发给右从臂；
- 右从臂由操作者直接拖动；
- 主臂只提供夹爪开合信号；
- 数据中的 7 个关节 action 来自右从臂自身状态；
- 主臂夹爪目标作为第 8 个 action 保存并控制右从臂夹爪。

本机物理右主臂接在驱动的左逻辑串口槽位 `DU0E1ZZP`。右臂单臂入口默认设置
`RIGHT_ARM_SWAP_TELEOP_ACTIONS=true`，只连接这个有信号的物理右主臂，并将其
夹爪字段映射到右从臂。若重新布线后物理右主臂改接右逻辑槽位，可显式关闭交叉映射：

```bash
RIGHT_ARM_SWAP_TELEOP_ACTIONS=false \
  scripts/run_right_arm_capture.sh
```

双臂模式仍使用其原有映射；`RIGHT_ARM_SWAP_TELEOP_ACTIONS` 只影响右臂单臂入口。

### 15.3 单臂启动和采集

先做不连接硬件的检查：

```bash
cd /home/kunpeng/lerobot-git/realman-x5-aligned-capture
scripts/run_right_arm_capture.sh --check
```

给本次数据集设置具体任务并正式启动：

```bash
TASK_RIGHT="Pick up the three pens on the table and place them into the container." \
  scripts/run_right_arm_capture.sh
```

这条任务已经是右臂单臂入口的默认值，因此也可以直接运行：

```bash
scripts/run_right_arm_capture.sh
```

启动日志应包含：

```text
[capture-runner] profile=right-arm-only control_mode=drag
[capture-app] task_index=right
[capture-app] force left=false right=true required=true
[capture-app] control mode=drag ... send_action=false send_gripper=true
```

硬件连接完成后，右主臂夹爪立即控制右从臂夹爪。等待网页状态变为 `READY`，把右从臂摆到希望每轮返回的任务起点，并用主臂把夹爪调到期望的初始开合状态，再踩最左踏板。程序会先记录关节和夹爪的实时状态，然后进入六维力拖动录制。成功时踩中踏板保存，失败时踩右踏板丢弃；两者都会在关闭拖动后只恢复右臂关节，夹爪继续跟随主臂，不需要再次对齐。当前双踏板软件急停已关闭，紧急情况请使用物理急停。

### 15.4 单臂数据格式

单臂数据集默认名称：

```text
kd_tacmae_force_right_<YYYYMMDD_HHMMSS>
```

当前右臂数据集预期包含：

| Feature | Shape | 内容 |
| --- | --- | --- |
| `action` | `[8]` | 右臂 7 关节 + 右夹爪 |
| `observation.state` | `[14]` | 右臂 7 关节 + 夹爪 + 六维力 |
| `observation.images.right_cam_right_wrist` | `[896, 896, 3]` | 去畸变右腕 RGB 视频 |
| `observation.depth_deformation.tactile_right_left` | `[3, 96, 128]` | 右夹爪左侧 X5 |
| `observation.depth_deformation.tactile_right_right` | `[3, 96, 128]` | 右夹爪右侧 X5 |

最终仍应以生成数据集的 `meta/info.json` 为准。

每个已保存 episode 的初始位姿还会单独写入：

```text
meta/episode_start_poses/episode-000000.json
meta/episode_start_poses/episode-000001.json
...
meta/inference_start_pose.json
```

每回合文件包含右臂 7 个关节角（度和弧度）、初始夹爪位置、任务文本、采集时间及复位结果。`meta/inference_start_pose.json` 是最近一次保存的起始状态快捷副本，推理启动程序可读取 `joint_positions_deg` 和 `gripper_position` 恢复机械臂及夹爪；夹爪范围为 0–1000。

### 15.5 单臂安全边界

默认配置中：

```bash
EPISODE_RIGHT_ARM_RESET_ENABLED=false
RIGHT_ARM_EPISODE_RETURN_TO_START_ENABLED=true
EPISODE_RETURN_TO_START_SPEED=5
EPISODE_RETURN_TO_START_MAX_DELTA_DEG=60
EPISODE_RETURN_TO_START_RESTORE_GRIPPER=false
READY_GRIPPER_FOLLOW_ENABLED=true
```

因此开始 episode 时不会先走到一组写死的关节角；开始踏板只读取当时的位置。保存或丢弃后会发生真实的右从臂运动，并沿关节空间返回这次记录的位置。路径不保证按笛卡尔直线运动，必须确保整个返回运动范围无人员、容器、桌沿、线缆等障碍物。左臂不会收到复位目标，主臂关节也不会驱动从臂。

程序不会把网页“停止程序”或 `Ctrl+C` 当成复位命令；遇到异常或危险时应停止并使用物理急停，而不是等待自动回位。动态回位只恢复右臂 7 个关节；夹爪初始位置仍会写入元数据，但不会自动执行夹爪复位。
