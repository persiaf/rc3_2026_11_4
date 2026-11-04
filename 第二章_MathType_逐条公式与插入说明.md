# 第二章 MathType 7 可复制公式代码（按原稿编号）

> 来源：用户上传《第二章 双臂移动机器人系统建模与问题描述》摘录。已提取 55 条带编号公式与 4 条未编号辅助表达式；原文声称共有 72 条，但本次粘贴文字中只实际提供 55 条带编号公式。
>
> 注意：下方编号只是查找标签，不属于粘贴内容。请仅复制每一个 latex 代码块内部的公式主体，不要复制编号，不要复制 ``` 这类围栏符号。输入 MathType 7 时不要粘贴 `\tag{}`、`\boxed{}`、`\[`、`\]`。复制到 Word 普通正文并使用「Toggle TeX」时，整段外加 `\[ ... \]`。
>
> 编辑兼容性：MathType 7 使用 Texvc 子集，复杂矩阵、对齐环境、中文字句可能因版本/设置转换失败。建议先用短式测试；复杂式可以插入矩阵/大括号模板后在 MathType 中编辑。以下已尽量规范化 LaTeX，但未在你的本机安装中逐条实测转换。

## Word 2019 + MathType 7 使用操作

1. Word 插入右侧自动编号公式：`Alt+Shift+Q`，在弹出的 MathType 编辑框中粘贴下方代码，检查转换后关闭窗口将公式返回 Word。
2. 直接打字：MathType 编辑器 `Preferences -> Workspace Preferences -> Allow TeX language entry from the keyboard` 勾选后，输入如 `\theta_b` 并 Enter。此选项是 TeX 输入模式，不是 Word 普通段落功能。
3. Word 正文切换法：直接输入 `\[ \theta_b \]`，选中或把光标放在同一行，点击 `MathType -> Toggle TeX` 或 `Alt+\`。复杂公式转换需自检。
4. 章号：在第二章首个公式之前，MathType -> Insert Chapter/Section Break，指定章号为 `2`；MathType -> Format Equation Numbers 设置类似 `(2-1)` 格式，然后使用右编号公式。不要在编辑框输入 `\tag{2-1}`。
5. 原文漏列的 17 个编号见末尾，不要依现有文件猜测公式内容。按 Word 的顺序连续插入才能维持编号一致。

## 公式代码


### 2.1.1 双臂移动机器人整体结构


**式（2-1）**

```latex
\boldsymbol{\xi} = \begin{bmatrix} x_b&y_b&\theta_b&h& \boldsymbol q_L^T& \boldsymbol q_R^T \end{bmatrix}^{T}
```


**式（2-2）**

```latex
n_\xi=4+n_L+n_R
```


### 2.2.1 世界坐标系与移动底盘坐标系


**式（2-3）**

```latex
\boldsymbol b= \begin{bmatrix} x_b&y_b&\theta_b \end{bmatrix}^{T}
```


**式（2-4）**

```latex
\boldsymbol R_z(\theta_b)= \begin{bmatrix} \cos\theta_b&-\sin\theta_b&0\\ \sin\theta_b&\cos\theta_b&0\\ 0&0&1 \end{bmatrix}
```


**式（2-5）**

```latex
{}^WT_B= \begin{bmatrix} \cos\theta_b&-\sin\theta_b&0&x_b\\ \sin\theta_b&\cos\theta_b&0&y_b\\ 0&0&1&z_b\\ 0&0&0&1 \end{bmatrix}
```


### 2.2.2 升降机构与左右机械臂坐标系


**式（2-6）**

```latex
{}^BT_H(h)=\begin{bmatrix} 1&0&0&0\\0&1&0&0\\0&0&1&h_0+h\\0&0&0&1 \end{bmatrix}
```


**式（2-7）**

```latex
{}^HT_{A_L}=\begin{bmatrix} \boldsymbol R_{A_L}&\begin{matrix} d_x\\d_a/2\\0\end{matrix}\\ \boldsymbol 0_{1\times3}&1\end{bmatrix}
```


**式（2-8）**

```latex
{}^HT_{A_R}=\begin{bmatrix} \boldsymbol R_{A_R}&\begin{matrix} d_x\\-d_a/2\\0\end{matrix}\\ \boldsymbol 0_{1\times3}&1\end{bmatrix}
```


### 2.2.3 多坐标系间齐次变换关系


**式（2-9）**

```latex
{}^WT_{A_\ell} = {}^WT_B \,{}^BT_H(h) \,{}^HT_{A_\ell}, \quad \ell\in\{L,R\}
```


**式（2-10）**

```latex
{}^WT_{E_\ell}(\boldsymbol{\xi}) = {}^WT_B \,{}^BT_H(h) \,{}^HT_{A_\ell} \,{}^{A_\ell}T_{E_\ell}(\boldsymbol q_\ell)
```


**式（2-11）**

```latex
\begin{bmatrix} \boldsymbol p_i^{A_\ell}\\1 \end{bmatrix} = \left({}^WT_{A_\ell}\right)^{-1} \begin{bmatrix} \boldsymbol p_i^W\\1 \end{bmatrix}
```


**式（2-12）**

```latex
\boldsymbol p_i^{A_\ell} = \boldsymbol R_{WA_\ell}^{T} \left( \boldsymbol p_i^W-\boldsymbol p_{WA_\ell} \right)
```


**式（2-13）**

```latex
{}^{A_\ell}T_{G_i} = \left({}^WT_{A_\ell}\right)^{-1}{}^WT_{G_i}
```


### 2.3.1 全向移动底盘运动学建模


**式（2-14）**

```latex
\boldsymbol v_B= \begin{bmatrix} v_x^B&v_y^B&\omega_b \end{bmatrix}^T
```


**式（2-15）**

```latex
\dot{\boldsymbol b}= \begin{bmatrix} \cos\theta_b&-\sin\theta_b&0\\ \sin\theta_b&\cos\theta_b&0\\ 0&0&1 \end{bmatrix} \boldsymbol v_B
```


**式（2-17）**

```latex
\begin{bmatrix} \omega_{FL}\\ \omega_{FR}\\ \omega_{RL}\\ \omega_{RR} \end{bmatrix} = \frac{1}{r_w} \begin{bmatrix} 1&-1&-a\\ 1&1&a\\ 1&1&-a\\ 1&-1&a \end{bmatrix} \begin{bmatrix} v_x^B\\ v_y^B\\ \omega_b \end{bmatrix}
```


**式（2-18）**

```latex
\begin{bmatrix} v_x^B\\ v_y^B\\ \omega_b \end{bmatrix} = \frac{r_w}{4} \begin{bmatrix} 1&1&1&1\\ -1&1&1&-1\\ -\frac1a&\frac1a&-\frac1a&\frac1a \end{bmatrix} \begin{bmatrix} \omega_{FL}\\ \omega_{FR}\\ \omega_{RL}\\ \omega_{RR} \end{bmatrix}
```


### 2.3.2 底盘几何轮廓与碰撞模型


**式（2-24）**

```latex
\mathcal F(\boldsymbol b) = \left\{ \begin{bmatrix}x_b\\y_b\end{bmatrix} + \boldsymbol R_2(\theta_b)\boldsymbol u \;\middle|\; \boldsymbol u\in\mathcal F_B \right\}
```


**式（2-25）**

```latex
\mathcal F_B= \left[-\frac{L_B}{2},\frac{L_B}{2}\right] \times \left[-\frac{W_B}{2},\frac{W_B}{2}\right]
```


**式（2-27）**

```latex
\mathcal F(\boldsymbol b)\cap\mathcal O = \varnothing
```


**式（2-28）**

```latex
\mathcal O_{d_s} = \mathcal O\oplus\mathbb B_{d_s}
```


**式（2-29）**

```latex
\mathcal F(\boldsymbol b) \cap\mathcal O_{d_s} = \varnothing
```


**式（2-31）**

```latex
\mathcal F(\gamma(s)) \cap\mathcal O_{d_s} = \varnothing,\quad \forall s\in[0,1]
```


### 2.4.1 双机械臂正运动学模型


**式（2-32）**

```latex
{}^{k-1}T_k= \begin{bmatrix} c_k&-s_kc_{\alpha_k}&s_ks_{\alpha_k}&a_kc_k\\ s_k&c_kc_{\alpha_k}&-c_ks_{\alpha_k}&a_ks_k\\ 0&s_{\alpha_k}&c_{\alpha_k}&d_k\\ 0&0&0&1 \end{bmatrix}
```


**式（2-33）**

```latex
{}^{A_\ell}T_{E_\ell}(\boldsymbol q_\ell) = \left( \prod_{k=1}^{n_\ell} {}^{k-1}T_k(q_{\ell,k}) \right) {}^{n_\ell}T_{E_\ell}
```


### 2.4.2 双机械臂逆运动学与工作空间约束


**式（2-35）**

```latex
{}^WT_{E_\ell} (\boldsymbol b,h,\boldsymbol q_\ell) = {}^WT_{G_i}
```


**式（2-36）**

```latex
\boldsymbol p_{E_\ell}^W (\boldsymbol b,h,\boldsymbol q_\ell) = \boldsymbol p_i^W
```


**式（2-37）**

```latex
q_{\ell,k}^{\min} \le q_{\ell,k} \le q_{\ell,k}^{\max}
```


**式（2-39）**

```latex
\mathcal W_\ell(\boldsymbol b,h) = \left\{ \boldsymbol p_{E_\ell}^W (\boldsymbol b,h,\boldsymbol q_\ell) \mid \boldsymbol q_\ell\in\mathcal Q_\ell \right\}
```


### 2.4.3 双机械臂微分运动学模型


**式（2-41）**

```latex
\boldsymbol v_{E_\ell} = \boldsymbol J_\ell(\boldsymbol q_\ell) \dot{\boldsymbol q}_\ell
```


**式（2-42）**

```latex
\boldsymbol J_{\ell,k}= \begin{bmatrix} \boldsymbol z_{k-1}\times (\boldsymbol o_E-\boldsymbol o_{k-1})\\ \boldsymbol z_{k-1} \end{bmatrix}
```


### 2.4.4 操作任务数学建模


**式（2-43）**

```latex
\mathcal T=\{t_1,t_2,\ldots,t_N\}
```


**式（2-44）**

```latex
t_i= \left( \boldsymbol p_i^W, \mathcal R_i, \kappa_i, j_i \right)
```


### 2.5.1 二维栅格环境模型


**式（2-46）**

```latex
M[u,v]= \begin{cases} 1,&\text{栅格被障碍物占据}\\ 0,&\text{栅格为自由空间} \end{cases}
```


**式（2-47）**

```latex
\begin{aligned} u&=\left\lfloor \frac{x-x_0}{r_g} \right\rfloor\\ v&=\left\lfloor \frac{y-y_0}{r_g} \right\rfloor \end{aligned}
```


**式（2-48）**

```latex
\begin{aligned} x_c&=x_0+\left(u+\frac12\right)r_g\\ y_c&=y_0+\left(v+\frac12\right)r_g \end{aligned}
```


### 2.5.2 多桌面工作区域模型


**式（2-50）**

```latex
\mathcal R= \{R_1,R_2,\ldots,R_{M_R}\}
```


**式（2-51）**

```latex
\mathcal T_j= \{t_i\in\mathcal T\mid j_i=j\}
```


### 2.5.3 静态与动态障碍物模型


**式（2-52）**

```latex
\mathcal O(t) = \mathcal O_s\cup\mathcal O_d(t)
```


**式（2-53）**

```latex
\mathcal C_{\mathrm{free}}(t) = \left\{ p\in SE(2) \;\middle|\; \mathcal F(p)\cap \left(\mathcal O(t)\oplus\mathbb B_{d_s}\right) = \varnothing \right\}
```


### 2.6.1 面向操作任务的可达停靠位姿生成问题


**式（2-54）**

```latex
p=(x,y,\theta)\in SE(2)
```


**式（2-55）**

```latex
\mathcal P_j= \{p_{j1},p_{j2},\ldots,p_{jK_j}\}
```


**式（2-56）**

```latex
\Omega_j = \Omega_{\mathrm{free}} \cap \Omega_{\mathrm{safe}} \cap \Omega_{\mathrm{reach},j} \cap \Omega_{\mathrm{orient},j}
```


**式（2-58）**

```latex
\rho(p,t_i) = \max\{ \rho_L(p,t_i), \rho_R(p,t_i) \}
```


**式（2-59）**

```latex
\mathcal B_i= \left\{ p\in\Omega_{j_i} \;\middle|\; \rho(p,t_i)\ge\rho_{\min} \right\}
```


### 2.6.2 多任务多停靠协同规划问题


**式（2-60）**

```latex
\mathcal C(p) = \{ t_i\in\mathcal T \mid \rho(p,t_i)\ge\rho_{\min} \}
```


**式（2-61）**

```latex
C_{ik}= \begin{cases} 1,&t_i\in\mathcal C(p_k)\\ 0,&t_i\notin\mathcal C(p_k) \end{cases}
```


**式（2-62）**

```latex
\begin{aligned} \sum_{k=1}^{K}x_{ik}&=1\\ x_{ik}&\le C_{ik}y_k\\ x_{ik},y_k&\in\{0,1\} \end{aligned}
```


**式（2-63）**

```latex
\bigcup_{p\in\mathcal S} \mathcal C(p) = \mathcal T
```


**式（2-66）**

```latex
L_{\mathrm{nav}} = L(\gamma_{0,\sigma_1}) + \sum_{j=1}^{Q-1} L(\gamma_{\sigma_j,\sigma_{j+1}})
```


**式（2-67）**

```latex
\begin{aligned} \min_{\boldsymbol X,\boldsymbol y,\sigma,\Gamma} \quad& \alpha\sum_{k=1}^{K}y_k + \beta L_{\mathrm{nav}} + \gamma J_{\mathrm{op}}\\ \mathrm{s.t.}\quad& \sum_{k=1}^{K}x_{ik}=1,\quad\forall i\\ & x_{ik}\le C_{ik}y_k,\quad\forall i,k\\ & x_{ik},y_k\in\{0,1\}\\ & \Gamma\subseteq\mathcal C_{\mathrm{free}} \end{aligned}
```


**式（2-68）**

```latex
\Pi_0= (\mathcal S,\boldsymbol X,\sigma,\Gamma)
```


### 2.6.3 环境变化下的执行方案在线调整问题


**式（2-69）**

```latex
\mathcal M_t \longrightarrow \mathcal M_{t+1}
```


**式（2-71）**

```latex
\begin{aligned} \min_{\Pi_{t+1}}\quad& \lambda_1 L_{\mathrm{rem}}(\Pi_{t+1})\\ &+ \lambda_2D_{\mathrm{edit}}(\Pi_{t+1},\Pi_t)\\ &+ \lambda_3T_{\mathrm{replan}}\\ \mathrm{s.t.}\quad& \mathcal T_{\mathrm{rem}}(t) \subseteq \bigcup_{p\in\mathcal S_{t+1}} \mathcal C_{t+1}(p)\\ & \Gamma_{t+1} \subseteq \mathcal C_{\mathrm{free}}(t+1) \end{aligned}
```


### 2.6.4 三类规划问题的统一关系


**式（2-72）**

```latex
(\mathcal M,\mathcal T,\boldsymbol\eta) \xrightarrow{\mathrm{P1}} (\mathcal P,\boldsymbol\rho,\boldsymbol C) \xrightarrow{\mathrm{P2}} \Pi_0 \xrightarrow{\mathrm{P3}} \Pi_{t+1}
```


## 原文未编号的辅助公式


2.4.1 双机械臂正运动学模型（原文第 208 行）

```latex
c_k=\cos\theta_k,\quad s_k=\sin\theta_k
```


2.4.1 双机械臂正运动学模型（原文第 210 行）

```latex
c_{\alpha_k}=\cos\alpha_k,\quad s_{\alpha_k}=\sin\alpha_k
```


2.4.1 双机械臂正运动学模型（原文第 222 行）

```latex
{}^WT_{E_\ell} = {}^WT_B \,{}^BT_H(h) \,{}^HT_{A_\ell} \,{}^{A_\ell}T_{E_\ell}(\boldsymbol q_\ell)
```


2.5.2 多桌面工作区域模型（原文第 309 行）

```latex
\mathcal T= \bigcup_{j=1}^{M_R}\mathcal T_j
```


## 本次源文未提供的带编号公式

（2-16）, （2-19）, （2-20）, （2-21）, （2-22）, （2-23）, （2-26）, （2-30）, （2-34）, （2-38）, （2-40）, （2-45）, （2-49）, （2-57）, （2-64）, （2-65）, （2-70）

请向原先的完整 `.tex` 或论文 Word 版本索取这些公式，不能依据缺失编号自行编造。
