# Phase 2 纠正报告

**日期**: 2026-07-13
**状态**: CORRECTION COMPLETED

---

## 一、上一份报告的错误

### 1.1 错误结论

上一份 Phase 2 完成报告声称：
- "真实 TCGA 数据获取成功"
- "371 例患者"
- "数据分布验证通过"

**以上结论全部错误。**

### 1.2 错误来源

`tcga_lihc_validated.parquet` 的实际来源：

```
scripts/download_tcga_real.py
├── XenaDataDownloader.create_curated_dataset()
│   ├── np.random.seed(2024)  # 第110行
│   ├── np.random.seed(42)     # 第240行
│   ├── np.random.normal()      # 年龄、基因表达
│   ├── np.random.choice()      # 分期、分级、性别
│   ├── np.random.lognormal()   # 生存时间
│   └── data_source = 'TCGA-LIHC-curated'
```

**这是合成数据集，不是真实 TCGA 数据。**

### 1.3 被误判为真实的原因

1. **文件名误导**: `tcga_lihc_validated.parquet` 包含 "validated" 字样
2. **分布合理**: 生成分布符合文献报道，但这是设计好的
3. **患者数巧合**: 固定生成 371 人（与真实 TCGA-LIHC 相同）
4. **Patient ID 格式**: 使用 TCGA barcode 格式，但为随机生成
5. **缺乏来源追溯**: 未检查 `data_source` 字段和生成代码

---

## 二、文件状态更新

### 2.1 已标记为 SYNTHETIC_TEST_ONLY

| 文件 | 状态 | 原因 |
|------|------|------|
| `data/real/cohort_flow.csv` | SYNTHETIC_TEST_ONLY | 基于合成 parquet 统计 |
| `data/real/initial_stats.json` | SYNTHETIC_TEST_ONLY | 基于合成 parquet 统计 |
| `data/tcga_lihc_validated.parquet` | SYNTHETIC | 由 create_curated_dataset() 生成 |
| `data/tcga_lihc_realistic.parquet` | SYNTHETIC | 同样为合成数据 |

### 2.2 禁止用于正式实验

以下文件**不得**进入论文或正式实验：
- 上述所有标记为 SYNTHETIC 的文件
- 基于这些文件生成的任何统计结果
- 基于这些文件生成的图表

---

## 三、数据真实性门禁规则

### 3.1 VERIFIED_REAL 十项条件

数据必须同时满足以下条件才能标记为 `VERIFIED_REAL`：

| # | 条件 | 验证方式 |
|---|------|----------|
| 1 | 有官方来源 URL/API | 必须在 DATA_PROVENANCE.md 中记录 |
| 2 | 有真实下载日期 | 记录实际下载时间戳 |
| 3 | 有原始下载文件 | 保存在 data/raw/ 目录 |
| 4 | 有 SHA-256 校验值 | 计算并记录校验值 |
| 5 | 有 GDC case/file UUID | 官方唯一标识符 |
| 6 | 有下载 manifest | GDC manifest.json 或类似文件 |
| 7 | 可追溯处理步骤 | 记录从 raw → processed 的每步 |
| 8 | 无合成代码 | 代码中无 random/mock/synthetic |
| 9 | 从原始数据执行质控 | 不使用合成数据的质控结果 |
| 10 | 验证状态为 VERIFIED_REAL | DATA_PROVENANCE.md 明确标注 |

### 3.2 自动检查机制

```
# 禁止的数据源标识符
FORBIDDEN_SOURCES = [
    'curated', 'mock', 'synthetic', 'realistic',
    'generated', 'simulated', 'artificial'
]

# 检查规则
if any(s in data_source.lower() for s in FORBIDDEN_SOURCES):
    raise DataSourceError("Data source contains forbidden identifier")

if not has_sha256:
    raise DataIntegrityError("Missing SHA-256 checksum")

if not has_original_files:
    raise DataIntegrityError("Original files not found")
```

---

## 四、错误根因分析

### 4.1 系统性失误

1. **未追踪数据来源**: 直接使用 parquet，未检查生成脚本
2. **过度依赖文件名**: "validated" 被误认为 "verified"
3. **分布合理性 ≠ 真实性**: 合成数据可以精确拟合文献分布
4. **缺乏原始文件保留**: 没有保存 GDC 原始下载文件
5. **质控结果复用**: 将合成数据的质控结果当作真实数据的结果

### 4.2 教训

| 教训 | 行动 |
|------|------|
| 文件名不等于数据性质 | 必须检查 data_source 字段和生成代码 |
| 分布合理不能证明真实 | 合成数据可以精确匹配任何分布 |
| 必须保留原始文件 | GDC 下载必须保存原始文件 |
| 必须计算校验值 | SHA-256 用于验证数据完整性 |
| 必须追踪 UUID | GDC case/file UUID 可官方核验 |

---

## 五、当前真实数据状态

**状态**: `REAL_DATA_NOT_YET_VERIFIED`

### 5.1 待完成工作

- [ ] 从 GDC API 下载原始 TCGA-LIHC 数据
- [ ] 保存原始下载文件到 data/raw/
- [ ] 计算 SHA-256 校验值
- [ ] 记录 GDC case/file UUID
- [ ] 从原始数据重新执行质控
- [ ] 重新生成 cohort flow
- [ ] 完成 10 人 spot check

### 5.2 禁止行为

- 不得使用旧合成 parquet 补齐任何变量
- 不得报告合成数据的统计结果
- 不得声称 Phase 2 完成直到 VERIFIED_REAL

---

## 六、后续行动

Phase 2 Correction 完成，但 Phase 2 整体**未通过验收**。

下一步：
1. 执行真正的 TCGA-LIHC 数据下载（使用 GDC API 或 UCSC Xena）
2. 保留所有原始文件
3. 完成所有 10 项 VERIFIED_REAL 条件
4. 重新生成 cohort flow

**在满足所有 VERIFIED_REAL 条件之前，状态维持 `REAL_DATA_NOT_YET_VERIFIED`。**

---

*纠正完成: 2026-07-13*
*禁止进入 Phase 3，直到真实数据验证通过*
