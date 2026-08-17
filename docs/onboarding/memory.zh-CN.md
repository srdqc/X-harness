# Memory 发布边界

Pico 保留 Memory Backend 协议和可选适配接口，但当前发布仓库不包含外部 Memory 实现、安装地址或配套制品。

首次配置请显式关闭 Memory：

```bash
pico onboard --skip-memory
```

有效配置是 `memory.backend = null`。Local Skills、Session、Context 与其他 Pico Runtime 能力仍然可用。

如果已有配置选择了一个未安装的 Memory Backend，Pico 会 fail closed，而不是静默退化。使用下面的命令检查并重置：

```bash
pico plugins
pico doctor --json
pico onboard --skip-memory --reset
```

不要根据源码中的 Adapter 名称猜测或寻找未发布的仓库、wheel 或下载地址。
