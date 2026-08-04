  1. 完善系统核心功能（当前重点）
  - 集成真实TCGA-LIHC数据加载
  - 完善PubMed文献检索
  - 增强风险评估模块

  2. 设计评估实验
  - 实现基线方法（Cox模型、DeepSurv）
  - 计算评估指标（C-index、AUC、校准曲线）
  - 决策曲线分析（DCA）

  2. 配置MCP PubMed - 设置真实的PubMed API访问


  3. 用户研究
  - 设计医生评估问卷
  - 收集评估结果

  4. 论文撰写
  - 按ACM TIST格式撰写
  - 重点突出：系统设计方法论 + 实证评估

    1. Real LLM Agent Evaluation - Run with actual API calls when ready:                                                                                                                python experiments/run_comprehensive_evaluation.py --agent --agent_samples 30 --no-mock                                                                                         
  2. Top metabolic risk factors identified (ready for paper Table):                                                                                                               
    - PKM, GLUD1, GPI, GLS, SCD (glycolysis/glutaminolysis genes)
    - AFP level and tumor stage (clinical factors)
  3. Figures to prepare:
    - Kaplan-Meier curves by risk groups
    - Calibration plots
    - Decision curve analysis
    - ROC curves at 1/3/5 years
  4. Remaining tasks from plan.md:
    - Doctor user study for report quality evaluation
    - Real TCGA-LIHC data integration via GDC API
    - MCP PubMed integration for literature access