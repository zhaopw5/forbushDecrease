# AMS 小时级质子通量研究方案（Markdown 版）

> 目标：围绕 **Forbush Decrease（FD）** 的**高时间分辨率动力学**、**CME vs CIR 驱动差异**与**刚度依赖的恢复时间常数**三条主线，建立一套可复现的数据处理与分析流程；

---

## 0. 数据与环境

### 0.1 必要数据

* **AMS 小时级质子通量**：按刚度/能段分箱（R≈1–80 GV），含时间戳与不确定度。
* **OMNI 1 h 太阳风/IMF**：`V, B, Bz, Np, Pdyn`，以及地磁指数 `Kp/Dst/AE`。
* **事件辅助（可选）**：CDAW CME 目录（用于事件溯源与 CME/CIR 粗分类）。



---

## 2. 预处理总则

### 2.1 时间对齐与清洗

* 以 **UTC 小时**为基准对齐 AMS 与 OMNI。
* 短缺口（≤2 h）仅做线性插值；更长缺口标记并在后续拟合中 **mask**。
* 对 OMNI 的明显尖峰（传输/标定伪影）做 3σ 裁剪但保留真实冲击跃迁。

### 2.2 基线与相对通量

对每个刚度 (R)，将通量 (\Phi_R(t)) 标准化为相对变化：
[
I_R(t)=\frac{\Phi_R(t)}{\langle \Phi_R\rangle_{t\in [t_0-48,h,,t_0)}},\qquad
\Delta I_R(t)=I_R(t)-1
]

* 其中 (t_0) 为事件 **onset**（后续由突变检测或冲击时刻给定）。
* 基线窗建议 24–48 h，必要时做灵敏度分析。

### 2.3 小时谱指数（可选）

用滑动窗口（7 个相邻刚度点）对 (\Phi(R,t)) 做加权最小二乘功率律拟合：
[
\Phi(R,t)\propto R^{-\Gamma(t)}
]
得到小时级 (\Gamma(t))（用于谱硬化/变软诊断与统计对比）。

---

## 3. 事件识别与相位划分

### 3.1 Onset/边界锚点

* **优先**：用 OMNI 的 **冲击/SSC** 或 (B, P_{\text{dyn}}) 的陡升沿作为 (t_0)。
* **无明确冲击**：对 (\Delta I_R(t)) 使用 **变点检测**（推荐 PELT）找显著负跳变；人工微调 1 次。

### 3.2 鞘层/磁云边界（(t_1)）

* 标准：鞘层后进入 **磁云**（低 (\beta)、低温、(B) 平滑旋转）；若温/β缺失，用 (B) 的平台/回落点近似。
* 无磁云事件（CIR）：可仅设下降相 + 慢恢复相，不强行贴“磁云”。


---

## 4. Idea 1｜FD 高时间分辨率动力学：双阶段/双时常数

### 4.1 分段指数模型（推荐基线模型）

* **下降相**（(t\ge t_0)）：
  [
  I_R(t)=1-A_R\left(1-e^{-(t-t_0)/\tau^{\downarrow}_R}\right)
  ]
* **恢复相**（(t\ge t_1)）：
  [
  I_R(t)=1-\Delta I_R(t_1),e^{-(t-t_1)/\tau^{\uparrow}_R}
  ]
* 其中 (A_R>0)、(\tau^{\downarrow}_R,\tau^{\uparrow}_R>0)。(t_0,t_1) 全刚度共享。

### 4.2 整体双指数（备选）

用两时常数在单一表达式中拟合快/慢过程（对部分事件可能更贴合）；以 AIC/BIC 与分段模型比较优劣。

### 4.3 参数估计与不确定度

* 非线性最小二乘（`scipy.curve_fit` 或 `lmfit`），参数加下界约束（≥0）。
* **区块自助法**（block bootstrap，块宽 6–12 h）评估时间相关噪声的不确定度。
* 结果产出：每个 (R) 的 ({A_R,\tau^{\downarrow}_R,\tau^{\uparrow}_R}) 及协方差。

### 4.4 刚度依赖与物理联系

* 画 (\tau^{\downarrow}_R) vs (R)、(\tau^{\uparrow}_R) vs (R)（log–log）；(\Delta I_R) vs (R)（降幅随刚度递减）。
* 物理解释：恢复相 (\tau^{\uparrow}_R \sim L^2/\kappa(R))，(\kappa \propto v,\lambda(R)/3)
  (\Rightarrow\ \lambda(R)\propto 1/\tau^{\uparrow}_R)（在固定几何尺度 (L) 近似下看 **相对刚度依赖**）。

### 4.5 与 OMNI 对齐的主图（论文图 1 原型）

* 面板 A：多刚度 (\Delta I_R(t)) + (B(t), V(t)) + 阴影标注（shock/sheath/MC）。
* 面板 B：拟合曲线与残差。
* 面板 C：(\tau^{\downarrow}_R,\tau^{\uparrow}_R) vs (R) 的幂律拟合。

---

## 5. Idea 2｜CME 驱动 vs CIR 驱动的差异统计

### 5.1 事件分群

* **CME-FD**：有冲击+鞘层+（常见）磁云；CDAW 可溯源为 Halo/宽 CME。
* **CIR-FD**：速度阶梯上升、波动增强、27 天复现、无典型磁云签名。
* 没有完备标签时，先用上述规则给出 **weak labels**，后续可辅以轻量分类器统一化。

### 5.2 指标与检验

每事件、每刚度 (R) 计算：

* 降幅 (A_R)、时常数 (\tau^{\downarrow}_R,\tau^{\uparrow}_R)；
* 谱指数变化 (\Delta \Gamma_{\max})、以及鞘层/磁云内的 (\Delta\Gamma)。
  **组间比较**：CME vs CIR 的 (A_R(R))、(\tau^{\uparrow}_R(R))、(\Delta\Gamma(R))。
  **统计检验**：Mann–Whitney U、Cliff’s delta（非参数，稳健）。

> **预期**：CME 组呈 **强刚度依赖**（低 R 降幅更大，(\tau^{\uparrow}_R) 斜率更陡）；CIR 组更“平坦/刚度无关”。

### 5.3 太阳风剖面耦合

* 事件级特征：鞘层 (\langle B\rangle,\ \sigma_B,\ \langle V\rangle)；磁云 (\langle B\rangle, \min(B_z))；sheath/MC 时长；上升沿陡峭度 (\max(dB/dt),\max(dV/dt))。
* 相关性与偏相关：(\Delta\Gamma) vs (\langle B\rangle_{\text{sheath}})、(A_R) vs (\max(dB/dt)) 等，控制事件强度等协变量。

> **机器学习练习 – 二分类（入门）**
>
> * **目标**：用 OMNI + AMS 事件级特征区分 CME vs CIR。
> * **模型**：Logistic Regression / Random Forest。
> * **输出**：交叉验证 AUC、混淆矩阵、特征重要度（或系数）图，用于“可解释性”。

---

## 6. Idea 3｜刚度依赖的恢复时间常数：反演扩散谱指数 (\delta)

### 6.1 幂律拟合

* 仅用恢复相：(\tau^{\uparrow}*R = C,R^{-\alpha})，对数线性回归求 (\hat{\alpha})（权重取 (\sigma*\tau^{-2})）。
* 事件层得到 (\hat{\alpha}_e\pm\sigma_e)。

### 6.2 总体合并与对比

* **随机效应模型**（或层级贝叶斯）合并 ({\hat{\alpha}_e})，得到总体 (\bar{\alpha}) 与异质性 (I^2)。
* 与 **Kolmogorov (1/3)**、**Iroshnikov–Kraichnan (1/2)** 对比，判断湍流谱更接近哪类。
* 条件分桶：按 (\langle B\rangle_{\text{sheath}}) 强弱、磁云时长、太阳周期位相，比较 (\alpha) 的分布差异。

> **机器学习练习 – 回归（入门）**
> 用太阳风/几何特征预测 (\alpha) 或某固定刚度的 (\tau^{\uparrow}_R)。
> 模型：Ridge/Lasso（可解释）或 XGBoost（效果好）。
> 可视化：偏依赖图（PDP）或 SHAP 值，说明“哪些环境变量让恢复更快/更慢”。

---

## 7. 关键代码骨架（Python；代码与注释用英文）

```python
# src/fd_models.py
import numpy as np

def decay_phase(t, t0, A, tau):
    # I(t) = 1 - A*(1 - exp(-(t - t0)/tau)) for t>=t0
    y = np.ones_like(t, dtype=float)
    m = t >= t0
    y[m] = 1.0 - A * (1.0 - np.exp(-(t[m] - t0) / tau))
    return y

def recovery_phase(t, t1, I1, tau):
    # I(t) = 1 - (1 - I1)*exp(-(t - t1)/tau) for t>=t1
    y = np.ones_like(t, dtype=float)
    m = t >= t1
    y[m] = 1.0 - (1.0 - I1) * np.exp(-(t[m] - t1) / tau)
    return y

def piecewise_fd(t, t0, t1, A, tau_d, tau_r):
    # Piecewise decay then recovery
    y = decay_phase(t, t0, A, tau_d)
    I1 = np.interp(t1, t, y)  # ensure continuity at t1
    yr = recovery_phase(t, t1, I1, tau_r)
    y[t >= t1] = yr[t >= t1]
    return y
```

```python
# src/fitting.py
import numpy as np
from scipy.optimize import curve_fit
from .fd_models import piecewise_fd

def fit_fd_piecewise(t_hours, Irel, t0, t1, p0=(0.05, 10.0, 24.0)):
    # f(t; A, tau_d, tau_r) with fixed t0, t1
    def fwrap(t, A, tau_d, tau_r):
        return piecewise_fd(t, t0, t1, A, tau_d, tau_r)
    bounds = ([0.0,  0.5,   1.0],   # A >= 0; taus > 0
              [0.9, 200.0, 400.0])
    popt, pcov = curve_fit(
        fwrap, t_hours, Irel, p0=p0, bounds=bounds, maxfev=20000
    )
    return popt, pcov  # A, tau_d, tau_r
```

```python
# src/stats_ml.py
import numpy as np
from sklearn.linear_model import LinearRegression

def fit_tau_powerlaw(Rs, taus):
    # Fit tau(R) = C * R^{-alpha}
    X = np.log(Rs).reshape(-1, 1)
    y = np.log(taus)
    reg = LinearRegression().fit(X, y)
    alpha = -reg.coef_[0]
    C = np.exp(reg.intercept_)
    return C, alpha
```

> 你只需在 `notebooks/10_fd_dynamics_single_event.ipynb` 中读入某一事件的 `t0, t1` 与多刚度的 (I_R(t))，循环调用 `fit_fd_piecewise` 并汇总表格/出图即可。

---

## 8. 图表与表格产出（可直接用于论文/答辩）

* **Fig. 1**（单事件展示）：
  A. (\Delta I_R(t))（多刚度） + (B,V) 对齐 + 相位阴影（shock/sheath/MC）
  B. 模型拟合 vs 观测 + 残差
  C. (\tau^{\downarrow}_R,\tau^{\uparrow}_R) vs (R)（log–log）与幂律拟合、置信区间

* **Fig. 2**（CME vs CIR 统计）：
  A. (A_R(R)) 的小提琴/箱线图（两组对比）
  B. (\tau^{\uparrow}_R(R)) 斜率（事件级 (\alpha) 的分布）
  C. (\Delta\Gamma(R)) 组间对比与显著性标注

* **Fig. 3**（物理对比）：
  (\bar{\alpha}) 的总体估计及 95% CI，与 **1/3、1/2** 的参照线对比；分桶（强/弱 (\langle B\rangle_{\text{sheath}})）的 (\alpha) 对比

* **表 1**：事件清单与 (t_0,t_1)、sheath/MC 时长、(A_R)、(\tau) 参数

* **表 2**：CME vs CIR 关键统计量（中位数、MAD、p 值、效应量）

---

## 9. 误差、稳健性与复现实验

* **误差**：拟合协方差 + block bootstrap；图中统一给 68% CI 带。
* **稳健性**：改变基线窗（24/36/48 h）、边界选择（±3 h）、缺口处理策略，比较参数漂移。
* **重现实验**：不同事件样本、不同太阳活动阶段（近极大 vs 近极小）重复验证。

---



## 11. 里程碑建议（2–4 周迭代）

* **Week 1**：完成 I/O 与预处理、单事件 quicklook、变点检测半自动化。
* **Week 2**：单事件分段指数拟合与 (\tau(R)) 幂律；生成 Fig. 1。
* **Week 3**：批量化跑多事件；初步 CME vs CIR 分组统计与 Fig. 2。
* **Week 4**：总体 (\bar{\alpha}) 估计与物理对比（Fig. 3）；误差/敏感性分析；撰写方法学小节。

---

## 12. 结论期待（写作要点）

* **Idea 1**：AMS 小时分辨率让下降/恢复两阶段的 (\tau) 得到清晰约束；(\tau^{\uparrow}_R) 随 (R) 呈幂律，和鞘层/磁云结构强相关。
* **Idea 2**：CME 事件的谱响应更“有刚度结构”，CIR 更“平坦”；差异与鞘层 (B) 强度、上升沿陡峭度显著相关。
* **Idea 3**：总体 (\bar{\alpha}) 与湍流谱指数（1/3 或 1/2）对比，指向日球湍流谱近似形态及其环境依赖。

---