  改动内容：

  - 把 E1/E2/E3 已完成结果写入：
      - /home/songt/MultiRM/RNA_modification_rep/main.tex
      - /home/songt/MultiRM/RNA_modification_rep/natcomm_unseen_transfer.tex
      - /home/songt/MultiRM/NATCOMM_MISSING_EXPERIMENTS.md

  - 新增并运行审计/解释脚本：
      - /home/songt/MultiRM/Scripts/natcomm_audit_interpretation.py

  - 生成补充产物：
      - Results/natcomm_audit/audit_summary.md
      - Results/natcomm_audit/lomo_config_audit.csv
      - Results/natcomm_audit/paired_significance.csv
      - Results/natcomm_audit/nucleotide_family_table.csv
      - Results/natcomm_audit/tanimoto_matrix.csv
      - Results/natcomm_audit/tanimoto_heatmap.png

  修正/补全点：

  - E4 leakage audit 已做：144 个 method/seed/held-out config 检查，Audit failures: 0。
  - E5 biological interpretation 已做：base-family 表、Tanimoto matrix、heatmap 已生成并写入 tex。
  - E6 manuscript restructuring 已补：NatComm 草稿不再说实验未完成，已加入三种子 ablation、paired stats、外部 RMBase stress test、audit 和解
    释。

  - 修了旧文档错误：paired_aucm_delta.csv 的旧写法改为实际存在的 paired_auc_delta.csv；去掉 “jobs submitted / not completed yet” 的过期表述。
  - 修了我新增 audit 脚本里的一个初版误报逻辑：无 pseudo-label source 时 held-out loss 不进入训练是预期行为，不应算失败。

  验证：

  - 运行了 python Scripts/natcomm_audit_interpretation.py。
  - 编译通过：
      - latexmk -pdf main.tex
      - latexmk -pdf natcomm_unseen_transfer.tex

  - 新 PDF 已生成：
      - /home/songt/MultiRM/RNA_modification_rep/main.pdf
      - /home/songt/MultiRM/RNA_modification_rep/natcomm_unseen_transfer.pdf

  剩余不是“实验没做”，而是结果限制：外部 RMBase 中 Psi 没有正例，m5U 和 m6Am 正例很少，所以外部验证只能写成 independent stress test，不能说 12
  类外部验证都充分覆盖。