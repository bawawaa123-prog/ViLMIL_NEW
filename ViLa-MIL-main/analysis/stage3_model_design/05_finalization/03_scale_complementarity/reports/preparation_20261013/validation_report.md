# Stage 3.5.3 准备与轻量验证记录

日期：2026-10-13（Asia/Shanghai）。解释器：`/opt/conda/envs/vila_mil_overlay_rt/bin/python`。

- Python compileall：通过。
- run_low_only_training.sh 的 bash -n：通过。
- CPU合成单元测试：11项通过（5.481秒）；预训练模型/Text加载使用stub，没有真实Slide推理或GPU任务。
- Stage351五折E0来源和Stage352完整35项来源检查：通过。
- Stage352汇总CSV的manifest哈希校验：通过；没有覆盖历史CSV或图表。
- 历史Low实际checkpoint、strict5、prototype shape、Router keys与选中Validation ACC/summary一致性已审计；结果见source_audit.json。
- train_low.py五折dry-run：通过，打印E0派生命令，没有模型构造或训练。
- run_independent_inference.py五折dry-run：通过，High来源Complete，Low来源全部Pending，没有模型构造或输出目录。
- 合成完整/局部报告检查：通过，局部数据不生成完整五折均值；真实ID仅进入private目录，公开准备表使用匿名token。
- 训练与推理完成标记检查：合成数据通过；改变训练协议、Warm Start、Fold、ID、标签、概率或hash的无效来源被拒绝。
- Hook检查：真实模型正常forward配合合成输入/stub预训练，Low/High两次cross_attention调用顺序和Patch维度正确，捕获注意力不改变概率；没有使用旧固定Dual logits的forward_with_attention输出。

未执行：正式Low五折训练、真实Low Validation五折推理、真实互补分组、真实G3/G2的Attention/Top Patch图像、专家病理复核。全部保持Pending。

正式训练根目录 `runs/low_matched_v1`、正式推理根目录 `runs/independent_val_v1` 均未创建。没有执行Git命令，没有修改核心模型、历史权重、Prompt、features、Mapping或strict5。
