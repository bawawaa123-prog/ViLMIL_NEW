# Yiyuan 严格五折 test 可视化

本报告读取上级目录的 fold_0.csv 至 fold_4.csv 和 timing_details.csv，使用 tools/generate_external_report_no_threshold.py 生成。
每个 fold 仅包含该模型的 test 集；五折 test 互斥，合计 968 张切片。
这里的 no_threshold_external_report 是沿用脚本默认目录名，本数据属于内部严格交叉验证测试。

配置与 eval_results/yiyuan/no_threshold_external_report 保持一致：plot_name=Yiyuan，bootstrap=1000，seed=2026，dpi=220。
预测直接使用 Y_hat；AUC 使用 p_1；F1 主指标使用 macro-F1。
标签 0=Adenocarcinoma，1=NonAdenocarcinoma；Sensitivity/Recall/Precision 以标签 1（非腺癌）为阳性类别。
95% CI 为按切片 bootstrap 的百分位区间。
best 按各折 test 准确率选出，仅用于展示；完整性能应结合五折结果解读。

复现命令（在 ViLa-MIL-main 下执行）：

```bash
conda run -n vila_mil_overlay_rt python tools/generate_external_report_no_threshold.py \
  --external_dir eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_eval \
  --plot_name Yiyuan \
  --negative_label Adenocarcinoma --positive_label NonAdenocarcinoma \
  --bootstrap 1000 --seed 2026 --dpi 220
```
