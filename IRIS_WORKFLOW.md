# IRIS + reliability + GCS

当前唯一的规划流程是：生成 IRIS 无碰撞多面体 → 与 SSM 包围盒可靠性分区求交 → 连通性筛选 → Drake GCS 求解 → 独立路径检查。网格区域生成和 fallback 最短路已删除。

## 已验证的状态

- Docker 镜像：`robotlocomotion/drake`，摘要已固定在 `Dockerfile.drake`。
- 容器内 Python：`/usr/bin/python3`，版本 3.12；Drake 版本 `0.0.20260601`。
- 当前缓存 `planar3R_iris_regions.npz`：6 个 IRIS 区域，成功区域生成累计 6.41 秒。该时间不包含 SSM、失败种子和进程启动。
- 当前生成参数：5 次迭代、100 次连续非碰撞优化尝试、configuration-space margin 0.02、连杆半径 0.02。
- 区域内部 hit-and-run 抽样：30,000 个构型，未检测到碰撞；结果在 `iris_region_validation.json`。这仍是数值抽样验证，不是连续无碰撞证明。
- `iris_path.json` 是前一组参数的成功路径：7 个原始区域 → 92 个可靠性区域 → 51 个实际求解区域，规划约 72.8 秒。路径检查了 935 个构型，端点误差约 `2.94e-13`。
- 前一组区域抽样曾发现 2 个碰撞点，已用加强参数重新生成并替换缓存。因此旧路径报告明确标记为历史参数结果；当前 6 区域缓存的完整 GCS 求解留待下一轮，不能把历史耗时当作当前参数的耗时。
- 按用户要求，网格测试已停止，不再做网格对比。

## 使用当前容器

本次配置的容器名为 `planar3r-iris-validation`，已停止，依赖保留在容器中，项目挂载到 `/workspace`。PowerShell 中执行：

```powershell
docker start planar3r-iris-validation
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_validate_iris.py
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_gcs_planner.py
docker exec planar3r-iris-validation /usr/bin/python3 -m unittest discover -s tests -v
docker stop planar3r-iris-validation
```

规划默认读取 IRIS 缓存、使用一组起终点候选、输出 `iris_path.json`。用 `--candidate-limit 3` 可尝试更多组合，耗时相应增加。`--sx/--sy/--gx/--gy` 修改工作空间端点；`--reliability-source start|goal|none` 选择可靠性分区来源。

恢复生成（复用已完成种子，不覆盖不匹配参数的缓存）：

```powershell
docker exec planar3r-iris-validation /usr/bin/python3 planar3R_drake_iris.py --resume
```

当前缓存已处理全部 22 个候选种子，所以只增加 `--max-regions` 不会生成更多区域；需要修改种子策略并选择新的输出文件。改变生成参数也应使用新的 `--output`。

## 从镜像重建环境

在项目根目录打开 PowerShell：

```powershell
docker build -f Dockerfile.drake -t planar3r-drake .
docker run --rm -it -v "${PWD}:/workspace" planar3r-drake
```

随后在容器中运行：

```sh
/usr/bin/python3 planar3R_gcs_planner.py
/usr/bin/python3 planar3R_validate_iris.py
```

Windows 的 PyCharm Python 3.10 装有另一个同名 `pydrake` 包，不能运行机器人 Drake；本工作流应使用 Docker。

## 实现说明

IRIS 保留原始半空间约束，直接叠加可靠性盒子的六个不等式；不会用外包盒替代斜面。空集和数值上扁平的交集通过 Chebyshev LP 去除。邻接允许区域重叠及边界接触。

候选构型先校正 SSM 的工作空间漂移，再筛选共有连通分量；每组起终点只求解相关的区域。GCS 原生 rounding 合并为一次调用，避免重复求解凸松弛。

机器人基座固定在世界坐标系，连杆碰撞几何覆盖完整长度；圆障碍物按连杆半径膨胀。当前模型支持项目中的单个圆障碍物。机械臂自碰撞被过滤，与原有解析检查器保持一致。

可靠性定义沿用每条 SSM 分支的关节角包围盒近似，关节可靠度为 0.7、0.5、0.3。工作空间背景可靠性图使用另一组可靠度，不能视为规划器实际使用的代价地图。

路径独立检查沿 GCS 的关节空间直线段进行，采样间距最多 0.01 rad；角度不进行最短方向 wrapping，以免改变优化出的路径。旧动画脚本仍使用硬编码的历史路点，尚未接入 JSON 路径，不能用它展示本次 IRIS 解。

参考实现位于 `D:/gcs-science-robotics/reproduction/bimanual/bimanual_iiwa_example.ipynb`、`reproduction/prm_comparison/prm_comparison.ipynb` 和 `gcs/linear.py`。当前 Drake 将旧 `IrisInConfigurationSpace` 接口改名为 `IrisNp`，生成脚本兼容两者。
