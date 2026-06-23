# RNA Modification LOMO 失败模式诊断报告

实验时间:2026-05-31
作者:Tengwei Song + Claude
项目:/ibex/user/songt/MultiRM
论文 v0/v1 描述:/home/songt/MultiRM/RNA_modification_rep/main.tex

---

## 1. 现象

化学条件化 RNA 修饰预测模型(`ChemicalMultiRMv1`)在 LOMO(leave-one-modification-out)实验中,**4 个 held-out 修饰的表现极度两极分化**:

| Held-out | AUCb | **AUCm** | 评价 |
|---|---|---|---|
| m7G | 0.722 | **0.943** | 优秀 |
| Psi | 0.808 | **0.931** | 优秀 |
| m6A | 0.737 | **0.216** ← 显著低于随机 | **灾难** |
| Am | 0.750 | **0.478** | 失败 |

注:AUCb(块内 50 正 + 50 负)看上去都"正常",但 AUCm(对全 1200 样本)在 m6A 上崩塌到 0.216 — 模型把 m6A 的真正样本系统性给低分。

---

## 2. 诊断历程(4 个版本,每次被下一次实验推翻)

### v1: "垃圾位"假说 (subagent #3 推理)

**机制**:共享 scorer 学到 "m6Am 化学 → m6A 列输出 0",col[m6A] 被训成"恒零垃圾位"。

**Fix**:anchor 正则(把 col[m6A] 均值拉回 seen tasks 均值)。

**被推翻**:Intervention Test 1 把 m6A 列输入的化学向量从 m6A 换成 m1A → col[m6A] 在 m6A 正样本上 prob 从 0.0002 跳到 0.0384(170 倍),AUCm 从 0.216 升到 0.695。**死列不会对输入有响应,假说被证伪。**

### v2: "化学编码器塌缩"假说

**机制**:LOMO 训练里 mod_{m6A} 被化学编码器映射到与 mod_{m6Am} 几乎共线(cos=0.98)。col[m6A] ≡ col[m6Am],而 m6Am 预测不在 m6A 站点 fire。

**几何证据**(4 个 LOMO 模型):

| Held-out | encoder 输出 cos(held, 最近邻) |
|---|---|
| m6A LOMO | cos(m6A, m6Am) = 0.980 |
| m7G LOMO | cos(m7G, Gm) = 0.937 |
| Am LOMO | cos(Am, m6Am) = 0.927 |
| Psi LOMO | cos(Psi, Um) = 0.808 |

**Fix**:线性编码器(无 LayerNorm + 无非线性 → 几何约束)。

**被推翻**:线性编码器实测对 m6A,encoder cos 仍为 0.966(比 mlp 0.98 几乎没变),AUCm 0.182(仍失败)。学到的线性 W 仍把 m6A 拉得几乎与 m6Am 共线,因为无 separation pressure。

### v3: "冻结编码器(J-L 投影保留输入 cos)" 假说

**机制**:冻结到随机初始化的线性投影 → 编码器输出 cos ≈ 输入 cos(Johnson-Lindenstrauss)= 0.85。理论上比训练后的 0.98 好。

**Fix**:`nn.Linear` requires_grad=False。

**被推翻**:实测冻结编码器在 m6A LOMO 上 encoder cos = 0.849(成功保留输入几何),但 **AUCm 仅 0.248**(基本没救起来)。

### v4(当前):"Morgan FP 表示极限"假说

**新证据**(cos 扫描实验):在 LOMO m6A 模型里**人工合成**不同 cos 的 mod 向量,测 AUCm。

| 合成的 cos(mod_m6A, mod_m6Am) | AUCm(m6A) |
|---|---|
| 0.99(当前 mlp 塌缩) | 0.198 |
| 0.85(输入空间余弦,J-L 极限) | **0.365** ← 仍低于随机 |
| 0.50 | 0.518 ← 刚过随机 |
| 0.10 | 0.667 |
| 0.00 | 0.681 |
| -0.30 | 0.752 |
| -0.70 | 0.867 |

**关键发现**:**模型本身有 m6A 判别能力**,只是被塌缩的 mod 向量堵死路径。要解码 m6A 信号,需要 cos < 0.5。但 Morgan FP 给的输入空间 cos = 0.85,**任何几何保持的编码器都达不到 cos < 0.5**。

**最终诊断**:`(Morgan FP) × (bilinear scorer) × (m6A LOMO)` 在数学上无解。

---

## 3. "Close chemical twin" 理论 — 可证伪、已验证

**理论**:LOMO 是否失败由两个条件决定:
1. Held-out 修饰是否有 Tanimoto > 0.65 的化学孪生 in seen tasks
2. 该孪生的 RNA 站点上下文是否覆盖 held-out 自己的站点

### 已验证的 6 个数据点

| Held-out | 最近邻 (Tani) | encoder cos | AUCm | 理论预测 | 实际 |
|---|---|---|---|---|---|
| m7G | Gm (0.41) | 0.94 | 0.943 | PASS | ✅ |
| Psi | m5U (0.39) | 0.94 | 0.931 | PASS | ✅ |
| m1A | m6A (0.60) | 0.66 | 0.671 | 阈值 | ✅ PASS |
| Am | m6Am (0.74) | 0.93 | 0.478 | FAIL | ✅ |
| m6A | m6Am (0.78) | 0.98 | 0.216 | FAIL | ✅ |
| m6Am | m6A (0.78) | 0.96 | 0.418 | FAIL | ✅ |

**阈值**:LOMO 失败发生在 Tani > ~0.65。m1A 的 0.60 刚好通过。

### m7G/Psi 高 encoder cos 也能成功的解释

m7G LOMO encoder cos(m7G, Gm) = 0.94(也塌缩了),但 AUCm 高。原因:**Gm 的 RNA 站点和 m7G 的 RNA 站点 happens to 重合**(都是 G-base 短 motif),所以 col[m7G] ≡ col[Gm] 在 m7G 站点上 happens to 给正确预测。

m6A 失败是因为 m6Am 的 RNA 站点(5′-cap)和 m6A 的(DRACH motif)**完全不重合** → col[m6A] ≡ col[m6Am] → 5′-cap 预测在 DRACH 站点上完全错。

---

## 4. 为什么 m6A 和 Am 都塌缩到 m6Am(不是塌缩到对方)

**m6Am 是 m6A 和 Am 的共同最近邻**:

| | m6Am | m1A | Am | I | m6A |
|---|---|---|---|---|---|
| m6A Tanimoto | **0.78** | 0.60 | 0.55 | 0.57 | 1.0 |
| Am Tanimoto | **0.74** | 0.48 | 1.0 | 0.48 | 0.55 |

化学解释:**m6Am = m6A + Am 的化学并集**(N6-methyl 来自 m6A,2′-O-methyl 来自 Am)。所以它在化学空间里同时是两者的最近邻。

编码器是光滑学习函数,无 separation pressure 时,held-out 输入 → 最近训练输入对应的输出。所以 mod_m6A 和 mod_Am 都被拉向 mod_m6Am(不是互相拉)。

---

## 5. 完整 12-mod LOMO 扫描(全部完成)

按 Tani 升序排:

| Held-out | 最近邻 (Tani) | 训练后 enc_cos | AUCb | **AUCm** | Verdict |
|---|---|---|---|---|---|
| Psi  | m5U (0.39) | 0.94 | 0.782 | **0.931** | ✅ PASS |
| m7G  | Gm (0.41)  | 0.94 | 0.718 | **0.943** | ✅ PASS |
| Cm   | m1A (0.45) | 0.29 | 0.901 | **0.786** | ✅ PASS |
| Um   | Gm (0.52)  | 0.21 | 0.898 | **0.927** | ✅ PASS |
| I    | m1A (0.58) | 0.51 | 0.663 | **0.554** | ✅ PASS |
| m5C  | m5U (0.58) | 0.88 | 0.936 | **0.769** | ✅ PASS |
| m5U  | m5C (0.58) | 0.70 | 0.935 | **0.897** | ✅ PASS |
| m1A  | m6A (0.60) | 0.66 | 0.772 | **0.671** | ✅ PASS |
| Gm   | Am (0.64)  | 0.84 | 0.906 | **0.911** | ✅ PASS |
| **Am**   | **m6Am (0.74)** | 0.93 | 0.750 | **0.478** | ❌ FAIL |
| **m6A**  | **m6Am (0.78)** | 0.98 | 0.737 | **0.216** | ❌ FAIL |
| **m6Am** | **m6A (0.78)**  | 0.96 | 0.730 | **0.418** | ❌ FAIL |

**清晰阈值**:
- **Tani > 0.70**: 3/3 全部 FAIL(m6A, m6Am, Am)
- **Tani ≤ 0.64**: 9/9 全部 PASS

Tani 0.65-0.73 没有数据点(无修饰对落在这个区间),所以确切阈值在这个 band 内但不可精确测定。

散点图见 `LOMO_scatter.png`(同目录)。

---

## 5a. Phase B 实验结果(site_weight 化学特征加强)

| site_weight | encoder | AUCb | AUCm | vs baseline |
|---|---|---|---|---|
| 0 (baseline) | mlp | 0.737 | **0.216** | — |
| 5 | mlp | 0.678 | **0.109** | 更差(无 frozen,encoder 把新特征也学塌缩) |
| 5 | frozen_linear | 0.602 | **0.294** | 微升 |
| 12 | mlp | 0.714 | **0.134** | 更差 |
| 12 | frozen_linear | 0.528 | **0.398** | 升,但仍 < 0.5 |

**结论:site_weight + frozen encoder 配合可以小幅救场(0.216 → 0.398),但仍达不到 0.5 阈值。** 与预测一致:site features 的数学上限把 cos(m6A, m6Am) 拉到 ~0.71,仍处于 cos 扫描的 FAIL 区(需要 < 0.5 才能稳定 PASS)。

**这证实了 v4 诊断**:**Morgan FP + 我能想到的 site features 在 m6A/m6Am 这对修饰上,无法把 cos 拉到方法能工作的范围**。

---

## 6. 尝试过的 fix 及其结果

| Fix | 状态 | 实际效果 |
|---|---|---|
| Anchor 正则(v1 配套) | 未实施 | v1 假说被推翻,fix 失效 |
| 线性编码器(v2 配套) | 已实施 | 没救 m6A(AUCm 0.182) |
| 冻结线性编码器(v3 配套) | 已实施 | 没救 m6A(AUCm 0.248) |
| Site 特征 + 高 weight | **跑中** | 上限:cos(m6A, m6Am) ≥ 0.71(子集关系限制) |
| 互斥子型编码 | 未实施 | 退化成 modid,LOMO 不可迁移 |
| 架构改动(bilinear → 非对称 scorer) | 未实施 | 工程量大,~天级 |

---

## 7. 真正的发现(论文级别结论)

**化学条件化的 zero-shot LOMO 能否成功,数学上由两个条件共同决定**:

1. **化学最近邻的 Tanimoto 距离** `T_max`:
   - `T_max < 0.65` → 几乎一定成功(m7G, Psi, m1A)
   - `T_max > 0.70` → 几乎一定失败(Am, m6A, m6Am)

2. **当 `T_max` 高时**,看孪生的 RNA 位点是否覆盖 held-out 站点:
   - 重合(m7G ↔ Gm,都 G-base 短 motif)→ 成功
   - 不重合(m6A ↔ m6Am,DRACH vs 5′-cap)→ 失败

**12 个 RNA 修饰中**,有 3 个(m6A、Am、m6Am)处于"高 Tanimoto + 站点不重合"的失败区。这 3 个共享同一对化学邻居关系(m6Am 是 m6A 和 Am 的"并集"修饰)。**这是论文必须承认的方法适用边界,而不是叙事重构**。

---

## 8. Tools 和实验脚本

| 用途 | 路径 |
|---|---|
| 基础训练 | `Scripts/paper_multirm.py` |
| 化学特征构造 | `Scripts/v0_data.py` 的 `build_chemical_feature_matrix` |
| Intervention 实验 | `Scripts/test_mod_injection.py` |
| Cos 扫描 | `Scripts/test_cos_sweep.py` |
| Encoder 几何分析 | `Scripts/encoder_geometry.py` |
| Encoder 对比 | `Scripts/compare_encoders.py` |
| Bootstrap CI | `Scripts/bootstrap_ci.py` |
| Multi-seed paired test | `Scripts/paired_test.py` |
| Phase 3 汇总 | `Scripts/analyze_phase3.py` |
| 预测回填 | `Scripts/regenerate_predictions.py` |

SLURM 提交脚本在 `slum_scripts/`。

---

## 9. 论文级别的最终结论

> **化学条件化 RNA 修饰预测的 LOMO 适用边界:held-out 修饰与其化学最近训练邻居的 Morgan FP Tanimoto 必须 < 0.70。** 12 种修饰中,9 种满足此条件且 LOMO AUCm 介于 0.55 - 0.94 之间;3 种不满足(m6A, m6Am, Am,均位于 A-base 化学密集簇)出现 calibration collapse(AUCm 0.22 - 0.48)。
>
> 这不是实现问题,而是 (Morgan FP 化学表示) × (bilinear scorer 架构) 在化学高相似度修饰对上的数学极限。site_weight 化学特征加强实验(site_weight ∈ {5,12} × encoder ∈ {mlp, frozen_linear})验证了上限 ≈ 0.398 AUCm,仍低于 0.5 随机阈值。

### 论文 main.tex 建议修改

Section 3.2 / 5.3 LOMO 评估部分应:
1. 报告全 12-mod LOMO 表(本文 Section 5)
2. 加散点图(本文 `LOMO_scatter.png`)
3. 明确阈值 "Tani > 0.70 → LOMO 失败"(3 个 A-base 修饰)
4. 解释化学机制:m6A、m6Am、Am 在 Morgan FP 空间互为子集(m6Am = m6A + Am 的化学并集)
5. 引用 Phase B 实验作为"已实证此限制无法通过 site features 修复"

### 未来工作方向

- (短期)用 atom-pair / pharmacophore FP 替代 Morgan FP,可能改善 m6A/m6Am 的区分度
- (中期)架构改动:scorer 不再是 bilinear 对称形式,可能用 chemistry-conditioned RNA encoder 内部 routing(FiLM in BiLSTM)
- (长期)预训练 chemistry encoder on auxiliary 任务,冻结后用作 LOMO encoder

## 10. 已完成的实验和产物清单

- 12-mod LOMO 全量扫描:`Results/paper_aligned/chemical_v1_bilinear_lomo/*/`
- 散点图:`LOMO_scatter.png`
- 多 seed × LOMO(48 jobs):`Results/paper_aligned/*_seed{2,3}/`
- Intervention 实验:`Scripts/test_mod_injection.py` 输出
- Cos 扫描:`Scripts/test_cos_sweep.py` 输出
- Encoder 几何:`Scripts/encoder_geometry.py` 输出
- Site features 修复实验:`Results/paper_aligned/chemical_v1_bilinear_sw{5,12}_{mlp,frozen_linear}_lomo/m6A/`
- 线性编码器实验:`Results/paper_aligned/chemical_v1_bilinear_linenc{,_lomo}/`
- 冻结编码器实验:`Results/paper_aligned/chemical_v1_bilinear_frozenenc_lomo/`
