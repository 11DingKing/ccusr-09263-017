# 非遗课程交流预约协调

本项目用于建设面向业务人员的纯服务端系统。代码按领域模型、应用服务、持久化与接口边界组织；时间、标识和外部输入应通过可替换端口接入，以便稳定复现状态变化。运行数据与本地配置不得写入源码目录。

## 架构

```
service_09252_008/
├── domain/            # 领域模型层
│   ├── models.py      #   课程包、导师、工坊资源、材料批次、接待窗口、预约、发运单、损耗、结算、事件
│   ├── rules.py       #   纯规则：前置培训、容量、安全等级、互斥资源、材料分配、运输周期
│   ├── forecasting.py #   容量需求预测纯逻辑：采样分桶、历史充足度评估、确定性模型、输入版本指纹
│   └── errors.py      #   领域错误（接口边界据此映射 HTTP 状态码）
├── application/       # 应用服务层
│   ├── ports.py       #   可替换端口：Clock / IdGenerator（测试注入手动时钟与序列 ID）
│   ├── catalog_service.py  # 目录登记与校验
│   ├── booking_service.py  # 预约状态机：申请/报价/锁定/改期/发运/到货/签到/结算/取消/恢复
│   └── capacity_forecast_service.py  # 容量需求预测：口径登记、解算、按输入版本重算
├── persistence/       # 持久化层
│   ├── store.py       #   存储端口 + 内存实现（快照回滚）
│   └── sqlite_store.py     # SQLite 实现（BEGIN IMMEDIATE，重启可恢复）
└── interfaces/
    └── http_api.py    # 接口边界：HTTP/JSON API（仅标准库）
```

## 领域规则要点

- **预约方案**：申请时校验前置培训（导师资格有效期须覆盖课程结束）、场地容量、
  材料安全等级（批次等级 ≤ 场地与窗口允许上限）、互斥资源（同资源或同互斥组时段不可重叠）、
  跨境运输周期（`now + lead_time ≤ slot_start`），并生成材料分配计划。
- **状态机**：`REQUESTED → QUOTED → LOCKED → SHIPPED → CHECKED_IN → SETTLED`，
  另有 `WAITLISTED / CANCELLED / EXPIRED`。窗口满或互斥被占时进入候补。
- **锁定**：在单事务内复查互斥并扣减库存，带 TTL；幂等键防止重复占位。
- **发运后不可移动**：`SHIPPED` 及之后的状态拒绝改期；取消时已发运材料记损耗
  （`cancel_after_shipment`），未发运预占回补库存，并按申请先后释放候补。
- **到货**：支持部分到货与在途损耗；发运单未关闭或到货不足时禁止签到。
- **结算**：按实际出勤折算消耗；国内余料退回库存，跨境余料记损耗
  （`non_returnable_leftover`），课中损坏记 `damaged_in_use`。
- **超时恢复**：过期锁定释放库存并晋级候补，过期报价退回待报价；
  服务启动时与 `POST /admin/recover` 均可触发。
- **时间**：内部一律 UTC；输入接受任意 ISO-8601 偏移（拒绝朴素时间）。

## 容量需求预测

资源规划口径：预测输入（采样窗口 + 模型参数）先登记落库，解算与重算都指向
明确的**输入版本**（范围 + 物化窗口 + 参数 + 观测点的规范化哈希）。

- **采样窗口**：半开区间按 `bucket_seconds` 切桶，聚合已确认预约
  （`LOCKED / SHIPPED / CHECKED_IN / SETTLED`）的席位需求；零需求桶同样是观测值。
  可选 `scope`（resource_id / window_id / institution）过滤口径。
- **模型参数**：`model_type`（`moving_average` / `linear_trend`）、
  `min_data_points`、`recommended_data_points`、`forecast_horizon_buckets`，
  另带模型代码版本 `capacity-forecast-v1`。
- **警告级别**：`info`（样本低于推荐覆盖，报告照出）→ `warning`（低于模型最小
  样本量，抑制报告）→ `error`（窗口内无任何历史，抑制报告）。警告携带
  code 与桶级证据，缺历史数据时**绝不写空报告**（只留运行记录供审计）。
- **重新计算**：必须显式携带 `input_version`；指纹与当前数据一致才重算
  （模型确定性，结果可复现），不一致返回 422 并给出请求版本与当前版本。

## 运行

```bash
python3 -m service_09252_008 --host 127.0.0.1 --port 8080
# 运行数据目录：--data-dir 或环境变量 SERVICE_09252_008_DATA_DIR
# （默认 ~/.local/state/service_09252_008，绝不写入源码目录）
```

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/packages` `/mentors` `/resources` `/material-batches` `/reception-windows` | 目录登记 |
| POST | `/bookings` | 申请（需幂等键） |
| POST | `/bookings/{id}/quote` | 报价 |
| POST | `/bookings/{id}/lock` | 锁定（需幂等键，可带 `ttl_seconds`） |
| POST | `/bookings/{id}/reschedule` | 改期（发运后拒绝） |
| POST | `/bookings/{id}/ship` | 发运（需幂等键） |
| POST | `/shipments/{id}/arrivals` | 到货（支持部分到货） |
| POST | `/shipments/{id}/losses` | 在途损耗登记 |
| POST | `/bookings/{id}/checkin` | 签到 |
| POST | `/bookings/{id}/settle` | 结算（`actual_attendance`、可选 `damaged`） |
| POST | `/bookings/{id}/cancel` | 取消（释放候补、按规则记损耗） |
| POST | `/admin/recover` | 恢复超时任务 |
| POST | `/capacity-forecasts` | 登记预测口径（采样窗口 + 模型参数） |
| GET  | `/capacity-forecasts/{id}` `/capacity-forecasts/{id}/runs` | 查询配置与运行记录 |
| POST | `/capacity-forecasts/{id}/run` | 解算一次预测（数据不足时返回警告、不写报告） |
| POST | `/capacity-forecasts/{id}/recompute` | 按显式 `input_version` 重算 |
| GET  | `/bookings/{id}` `/health` | 查询 |

幂等键经请求头 `Idempotency-Key` 或载荷字段 `idempotency_key` 传入；
同键重放返回首次结果（`idempotent_replay: true`），同键不同载荷返回 409。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

覆盖：主流程端到端、前置培训/容量/安全/互斥/运输周期规则、跨时区、
幂等重放、并发锁定（内存与 SQLite 双后端）、重启后超时恢复、
部分到货与在途损耗、取消释放候补与损耗记录、
容量需求预测（口径持久化、显式输入版本重算、缺历史数据不写空报告、SQLite 重启恢复）、
HTTP 接口边界。

## 编译检查

```bash
python3 -m compileall -q service_09252_008 tests
```

扩展模块覆盖证据、审批、权限、留存、对账与恢复等业务边界。
