# Planar 3R reliability-aware motion planning

平面三关节机械臂的可靠性运动规划实验。当前规划工作流为：

```text
IRIS 无碰撞多面体 → 可靠性分区求交 → 连通性筛选 → GCS 路径 → 独立碰撞/端点检查
```

默认连杆长度为 `[1,1,1]`，起点 `(0.5,0.5)`，终点 `(1.5,1.5)`；障碍物为中心 `(0.75,0.75)`、半径 `0.15` 的圆。规划器使用的关节可靠度为 `0.7、0.5、0.3`。

## 主要文件

| 文件 | 用途 |
| --- | --- |
| `planar3R_drake_iris.py` | IRIS 区域生成及断点续跑 |
| `planar3R_iris_regions.py` | 缓存读写、可靠性求交和多面体邻接 |
| `planar3R_gcs_planner.py` | 主入口、场景及可靠性定义、JSON 输出 |
| `planar3R_iris_gcs_planner.py` | GCS 求解、候选筛选和路径验证 |
| `planar3R_validate_iris.py` | 区域内部独立抽样验证 |
| `planar3R_collision.py` | 正运动学连杆几何及解析碰撞检查 |
| `planar3R_two_point_ssm_views.py` | IK、SSM 跟踪、端点校正和绘图 |
| `planar3R_bbox_reliability_partition.py` | SSM 包围盒可靠性分区及绘图 |
| `gcs/` | Drake GCS 封装 |
| `Dockerfile.drake` | 可重建的 Docker 运行环境 |
| `tests/test_iris_regions.py` | 几何、缓存、候选选择和 Drake 集成测试 |

其他可靠性地图与动画脚本保留作独立研究/历史展示。网格路径规划和 fallback 已删除。

## 当前状态和运行

详见 [IRIS_WORKFLOW.md](IRIS_WORKFLOW.md)。当前缓存包含 6 个加强参数生成的 IRIS 区域，30,000 个内部抽样未检测到碰撞。`iris_path.json` 是旧参数的成功路径报告，已标记；新缓存的完整规划待下一轮运行。

在已配置的容器中：

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_gcs_planner.py
docker stop planar3r-iris-validation
```

IRIS 的碰撞认证和独立检查均为数值方法；未检测到碰撞不代表严格连续证明。当前模型不检查机械臂自碰撞。
