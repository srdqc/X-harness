# Onboarding media manifest

Screenshots, GIFs, videos, and report assets stay outside Git. This manifest defines reproducible scenes without committing binaries.

| ID | Scene | Required visible proof | Secret treatment |
| --- | --- | --- | --- |
| `first-turn` | `pico onboard --skip-memory` through the first reply | Provider selected and an `Agent:` reply | API key masked; disposable repository path |
| `feishu-config` | Feishu Open Platform plus Pico CLI | long connection, `im.message.receive_v1`, published version, redacted channel config | all application secrets and tenant identifiers redacted |
| `feishu-live` | one inbound message and Pico reply | accepted inbound event and reply in the same conversation | user names, open IDs, message IDs, and credentials redacted |
| `agent-install` | released Pico installation and health check | `pico doctor --json` | signed URL query strings and local home paths redacted |

Use disposable credentials, bind each capture to a Pico tag and operating system, and never label a fixture or skipped probe as a live success.
