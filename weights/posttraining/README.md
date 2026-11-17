# 可检查的后训练适配器

这里保存 `configs/posttraining/student_frozen.json` 对应的 LoRA 适配器，不包含基础模型。固定基础模型为 `mlx-community/Qwen2.5-1.5B-Instruct-4bit`，revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`。

- `adapters.safetensors`：冻结的实际训练权重。
- `adapter_config.json`：加载所需的层数、LoRA 参数和训练类型。
- `LICENSE_QWEN.txt`：上游 Qwen 基础模型的 Apache 2.0 许可副本；来源版本记在 `base_model_license_source.json`。

训练参数、数据统计、训练日志摘要与适配器 SHA256 见 `records/posttraining/training_summary.json`。冻结配置会验证权重摘要；原始基座权重单独下载，推理时同时加载基础模型和本目录适配器。生成式训练样本、课程问答和逐题教师轨迹不在本目录中。

加载与完整运行方法见 [后训练说明](../../docs/POSTTRAINING.md)。在项目根目录可以直接验证加载：

```python
from mlx_lm import load
model, tokenizer = load("models/student", adapter_path="weights/posttraining")
```
