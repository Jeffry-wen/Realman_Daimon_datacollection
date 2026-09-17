# 右臂单臂数据格式、触觉解码与部署约定

本文记录右臂单臂采集数据的实际语义，核对对象为本机数据集
`kd_tacmae_force_right_20260917_174110`（50 episodes，30 FPS）以及当前
`config/lab_5090.env`。数据集的 `meta/info.json` 已确认 action 为 8 维、
state 为 14 维，两路触觉为 FFV1 `gbrp16le` 的三通道 uint16 MOV。

代码来源分为两部分：

- 采集工程：本仓库 `scripts/capture_realman_x5_force_aligned_app.py` 和
  `scripts/capture_realman_x5_force_app.py`。
- 兼容 LeRobot：`$LEROBOT_ROOT`，本机实际为
  `/home/kunpeng/lerobot-git/lerobot-0.5.1-main-20260520-sunday`。

## 1. 夹爪 action 的含义与转换

### 字段定义

右臂单臂数据的顺序是：

```text
action[0:7] = right_main_joint1 ... right_main_joint7
action[7]   = right_main_gripper

observation.state[0:7] = right_main_joint1 ... right_main_joint7
observation.state[7]   = right_main_gripper
observation.state[8:14] = Fx, Fy, Fz, Mx, My, Mz
```

```text
action[7] 表示：从臂目标硬件位置 / 1000。
它不是主臂原始夹爪值。

observation.state[7]：是从臂实际夹爪位置 / 1000。
```

这批数据采集时 `SAVE_GRIPPER_ACTION_AS_TARGET` 没有在 env 中显式填写，
但程序默认值及实际生效值均为 `true`。

### 采集时从主臂到 action/硬件的完整公式

令主臂原始开合值为 `leader`，实际参数为：

```text
leader_min = 0.066
leader_max = 0.971
gain       = 1.1
软件允许的硬件位置 = 100 .. 1000
```

转换为：

```python
normalized = (leader - 0.066) / (0.971 - 0.066)
normalized = 0.5 + (normalized - 0.5) * 1.1
normalized = clip(normalized, 0.0, 1.0)

# 本批次安全限制的等价效果
normalized = clip(normalized, 0.1, 1.0)

action_7 = normalized
hardware_target = int(normalized * 1000)
```

`LingkongGripperV2Wrapper` 对新夹爪采用同向映射，没有 `1-normalized`
反向操作。硬件/协议本身接受 `0..1000`，但本批采集的安全层进一步限制为
`100..1000`。

```text
硬件位置 0：完全闭合/夹紧方向（本批安全限制不会下发到 0）
硬件位置 1000：完全打开方向
硬件协议范围：0..1000
本批实际允许范围：100..1000
```

其他生效规则：

- gain：`1.1`，以归一化中点 0.5 为中心放大。
- 反向：无。
- 截断：先截到 `[0,1]`，再由安全层限制到 `[0.1,1]`。
- 硬件命令死区：主臂原始值变化小于 `0.01` 时可以不重复下发；这约等于
  归一化目标变化 `0.01215`，但保存的 action 本身仍可记录更小变化。
- 夹爪命令最大发送频率：50 Hz；本批 action 生成/保存频率为 30 Hz。
- 夹爪命令由异步线程下发，因此 `action[7]` 是目标值，不是硬件到位回执；
  到位结果应看 `observation.state[7]`。

### 部署时的正确转换

模型输出已经是从臂目标坐标，推荐直接下发：

```python
a = float(action[7])
target = int(clip(a, 0.1, 1.0) * 1000)
gripper.move_to_pos(target)
```

不要把数据集的 `action[7]` 直接传给当前的 `move_from_leader()` 或把它当成
主臂原始值，否则会重复应用 min/max/gain 映射。

如果部署代码必须经过当前 `robot.send_action()` 的主臂坐标接口，应先做逆变换：

```python
a = clip(float(action[7]), 0.1, 1.0)
normalized_leader = 0.5 + (a - 0.5) / 1.1
leader_value = 0.066 + normalized_leader * (0.971 - 0.066)
# 再把 leader_value 交给现有 move_from_leader 路径
```

### 打开、闭合示例

```text
完全打开命令：
  action[7] = 1.0
  硬件目标 = 1000
  对应逆变换后的主臂接口值约为 0.929864（更大的值也会饱和到 1000）

本批允许的最闭合命令：
  action[7] = 0.1
  硬件目标 = 100
  对应逆变换后的主臂接口值约为 0.189409（更小的值会被安全层限制到 100）
```

## 2. 戴盟三通道的保存与还原规则

### 两路数据和实际安装位置

```text
observation.depth_deformation.tactile_right_left：
  右从臂夹爪的左侧手指/左侧触觉片，夹爪内侧接触面。

observation.depth_deformation.tactile_right_right：
  右从臂夹爪的右侧手指/右侧触觉片，夹爪内侧接触面。
```

这里的 left/right 是本项目对右夹爪两个 pad 的物理命名。传感器像素坐标相对
机器人/桌面的精确朝向尚未形成标定文件，因此该项为“未知”；不能仅根据名称推断
像素 `+x/+y` 的空间方向。

### MOV 解码后的通道顺序

保存前 HWC 顺序和 LeRobot 解码后的 CHW 顺序都是：

```text
channel 0：depth
channel 1：deformation_x
channel 2：deformation_y
```

LeRobot 解码器默认返回 `[3, 96, 128]` 的 float32 张量，并把 uint16 除以
`65535` 缩放到 `[0,1]`。如果直接使用 PyAV 解码 `gbrp16le`，可得到
`[96,128,3]` 的 uint16。

### 编码公式

SDK 接口为 `sensor.getDepth()` 和 `sensor.getDeformation2D()`。采集代码使用：

```python
depth_u16 = round(depth * 1000.0)
dx_u16 = round(deformation_x * 1000.0 + 30000.0)
dy_u16 = round(deformation_y * 1000.0 + 30000.0)

depth_u16 = clip(depth_u16, 0, 65535)
dx_u16 = clip(dx_u16, 0, 65535)
dy_u16 = clip(dy_u16, 0, 65535)
```

NaN 被替换为零；正负无穷按 uint16 可表达边界替换，然后再截断。编码后如果
SDK 原图不是 `96x128`，使用 OpenCV `INTER_AREA` 在 uint16 空间缩放到
`96x128`。随后使用 FFV1 + `gbrp16le` 无损保存到 MOV。

### 还原公式

若使用 LeRobot 的视频解码函数，先恢复 uint16：

```python
u16 = round(decoded_float_chw * 65535.0)

depth = u16[0] / 1000.0
deformation_x = (u16[1] - 30000.0) / 1000.0
deformation_y = (u16[2] - 30000.0) / 1000.0
```

若使用 PyAV 直接得到 HWC uint16：

```python
depth = frame[..., 0].astype(float) / 1000.0
deformation_x = (frame[..., 1].astype(float) - 30000.0) / 1000.0
deformation_y = (frame[..., 2].astype(float) - 30000.0) / 1000.0
```

因此，已确认形变通道必须使用 `(raw - 30000) / 1000`，不能把 30000 当成
真实形变量，也不能只做 `/65535` 后直接用于物理解释。

### 单位、零点及空间处理

```text
depth 单位：戴盟 SDK 算法输出单位；物理单位未知。
deformation_x/y 单位：戴盟 SDK 算法输出单位；物理单位未知。

算法零值的保存参考：
  depth = 0              -> uint16 0
  deformation_x/y = 0   -> uint16 30000
```

- 采集层没有减去启动基准帧或每 episode 基准帧。
- 戴盟 SDK 内部是否已经做参考面/零接触标定：未知（厂商内部实现不可见）。
- 无接触时并不保证所有像素严格等于 `0/30000/30000`，应保留实际静态基线。
- 编码会对超出 uint16 可表达范围的值饱和：depth 下限为 0，因此负 depth 会被
  截为 0；形变可保存的范围约为 `[-30.000, 35.535]` 算法单位。
- 保存路径没有通道交换、旋转、镜像或几何裁剪。
- 可能发生 `INTER_AREA` 缩放到 `96x128`。
- 网页预览中的顺时针旋转、伪彩色和阈值只影响预览，不影响保存的 MOV。

### 编解码示例

假设 SDK 某像素输出：

```text
depth = 1.234
deformation_x = -0.250
deformation_y = +0.400
```

保存及读回为：

```text
保存 uint16       = [1234, 29750, 30400]
LeRobot 解码浮点  = [0.01882963, 0.45395590, 0.46387427]
按上述公式还原    = [1.234, -0.250, +0.400]
```

实际数据集首个触觉视频中的一个像素也已抽查：保存值
`[0, 30011, 30000]`，还原为 `[0.000, 0.011, 0.000]`。

## 3. 关节与动作时间规则

```text
实际采集模式：drag
拖动类型：六维力精准拖动，mode=3，同时拖动位置和姿态

关节顺序：
  right_main_joint1
  right_main_joint2
  right_main_joint3
  right_main_joint4
  right_main_joint5
  right_main_joint6
  right_main_joint7

异步 state reader 对采集层输出单位：弧度
配置 use_degrees：false
```

drag 模式把同一目标帧选择到的从臂实际关节 state 再复制到关节 action，所以这批
数据中同帧 `action[0:7] == observation.state[0:7]`。

第 t 帧夹爪 action 是与第 t 帧目标时间最近的当前控制命令，不是“下一控制周期”
标签。采集器以 30 Hz 建立目标时间轴，在 action 环形缓存中做 nearest-neighbor
选择；没有 action 插值和平滑。允许的 action 最大时间差为 250 ms，超过后在当前
fail-closed 配置下中止该 episode。

时间戳细节：

- action 循环先读取主臂命令，再在发送硬件前确定 `sample_t`。
- drag 模式会把 `sample_t` 改为本次采用的从臂 state 样本时间，因此夹爪命令沿用
  该 state 时间，而不是单独记录主臂夹爪读取时刻。
- 随后调用 `robot.send_action()`，最后把目标 action 放入缓存。
- 数据集只保存帧时间戳，没有单独保存夹爪发送前/发送后的硬件时间戳；
  `RECORD_SYNC_DIAGNOSTICS=false`，所以本批数据也没有保存 action offset 诊断列。
- `ALIGNED_WRITER_DELAY_S=0.45` 是为了等待异步源到齐后再选择目标时刻附近的数据，
  不是把训练标签定义成未来 0.45 秒动作。

部署关节轨迹时：

```text
LeRobot action 输入：7 个绝对关节位置，单位为弧度。
驱动内部：转换为度。
默认真机接口：rm_movej_canfd(target_degrees, False, 0)。
接口语义：绝对关节目标，不是增量。
建议发送频率：30 Hz，与本批数据 FPS 一致。
```

如果切换 `REALMAN_JOINT_COMMAND_MODE=movej`，则会调用阻塞/非阻塞的
`rm_movej`，不再是本段描述的默认 CAN-FD 流式接口。

## 4. 腕部六维力定义

右臂单臂 state 中六维力位于索引 8..13：

```text
observation.state[8]  = right_force_sensor_fx
observation.state[9]  = right_force_sensor_fy
observation.state[10] = right_force_sensor_fz
observation.state[11] = right_force_sensor_mx
observation.state[12] = right_force_sensor_my
observation.state[13] = right_force_sensor_mz
```

```text
六个通道顺序：Fx, Fy, Fz, Mx, My, Mz
力单位：N
力矩单位：N·m
实际字段：work_zero_force_data
坐标系：RealMan 当前工作坐标系
```

轴方向是控制器当前工作坐标系的 `+X/+Y/+Z`，符合右手定则。当前工作坐标系相对
机器人基座、桌面和相机的具体设置没有随 LeRobot 数据集保存，因此精确物理朝向为
“未知”，不能在未经现场查询的情况下直接宣称等同于基座/世界坐标系。

仅作为硬件参考，RealMan SDK 对传感器原始坐标系的说明是：传感器正上方为 `+Z`，
航插反方向为 `+Y`，`+X` 由右手定则确定；但本批保存的是
`work_zero_force_data`，不是传感器坐标系的 `zero_force_data`。

清零和补偿规则：

- 连接时清零：`true`。
- 每个 episode 开始时清零：`true`。
- 顺序为：开始命令被接受、开启六维力拖动、执行 `rm_clear_force_data()`、等待
  0.2 秒，然后启动正式时间轴。
- 清零姿态和夹爪状态：操作者踩下开始踏板时的任务起始姿态和实时夹爪状态。
- 是否接触物体：代码不检测，实际情况未知；操作要求应为空载且不接触物体，否则
  接触力会被当作零点扣除。
- 采集代码没有自行做重力、工具负载补偿；`work_zero_force_data` 是 RealMan
  控制器给出的“当前工作坐标系下系统外受力”。控制器内部使用了哪些已配置的
  重心/工具负载参数，本批数据元信息中没有保存，因此为“未知”。
- 采集代码没有对六维力做低通、均值或中值滤波。读取线程以 100 Hz 轮询并缓存最新
  值，30 Hz 对齐器选择目标帧附近的 state；RealMan 控制器内部滤波方式未知。

## 5. 本批数据实际生效配置

```text
FPS=30
CAPTURE_CONTROL_MODE=drag
SAVE_GRIPPER_ACTION_AS_TARGET=true        # 程序默认值，env 未显式写出

leader_gripper_min=0.066
leader_gripper_max=0.971
gripper_gain=1.1
GRIPPER_MIN_POSITION=100
GRIPPER_MAX_POSITION=1000
GRIPPER_ACTION_SEND_HZ=50
GRIPPER_ACTION_DEADBAND=0.01              # 程序默认值
GRIPPER_SPEED=60

use_degrees=false
默认关节下发接口=rm_movej_canfd
ALIGNED_ACTION_HZ=30                      # 未显式设置，继承 FPS

FORCE_SENSOR_DATA_KEY=work_zero_force_data
FORCE_SENSOR_READ_HZ=100
FORCE_SENSOR_CLEAR_ON_CONNECT=true
FORCE_SENSOR_CLEAR_ON_EPISODE_START=true
FORCE_SENSOR_CLEAR_SETTLE_S=0.2

WRIST_TIMESTAMP_MODE=frame_clock
WRIST_SOURCE_LATENCY_FRAMES=5
WRIST_SOURCE_LATENCY_S=0.0
ALIGNED_WRITER_DELAY_S=0.45
ALIGNED_TACTILE_USE_CACHE_HISTORY=true

X5_TACTILE_MODE=standard
X5_TACTILE_MAX_FPS=45
X5_TACTILE_ASYNC_CACHE_FPS=120
TACTILE_WIDTH=128
TACTILE_HEIGHT=96
```

## 6. 关键实现位置

```text
采集工程：
scripts/capture_realman_x5_force_aligned_app.py
  GripperSafetyLimits
  replace_joint_actions_with_follower_state
  ActionLoop
  _collect_aligned_action
  capture_episode_aligned

$LEROBOT_ROOT/tools/bi_x5_capture_app.py
  map_gripper_actions_for_dataset

$LEROBOT_ROOT/src/lerobot/robots/realman_ugripper_notac_new/lingkong_gripper.py
  LingkongGripperV2Wrapper.move_from_leader

$LEROBOT_ROOT/src/lerobot/robots/bi_realman_ugripper_notac_new/x5_tactile_flux_receiver.py
  _legacy_depth_to_uint16
  _legacy_vector_to_uint16
  X5TactileFluxReceiver.read_images

$LEROBOT_ROOT/src/lerobot/datasets/video_utils.py
  decode_tactile_video_frames_pyav
  decode_tactile_video_frames_torchcodec

$LEROBOT_ROOT/src/lerobot/robots/realman_ugripper_notac/realman_ugripper_notac.py
  AsyncForceSensorReader
  RealmanUGripperNotac.send_action
```
