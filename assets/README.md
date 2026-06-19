# assets/

Demo media + diagrams for AgentFuse.

| file | what it is | committed? |
| --- | --- | --- |
| `atlas-light.svg` / `atlas-dark.svg` | architecture / data-flow diagram (light + dark) embedded in the README | yes |
| `demo.gif` | the recorded terminal GIF embedded in the README | no — generated |

The vhs script that records the demo lives at [`docs/demo.tape`](../docs/demo.tape).

## Regenerate the GIF

```bash
# install vhs once: https://github.com/charmbracelet/vhs
vhs docs/demo.tape   # writes assets/demo.gif
```

The demo runs fully offline (litellm `mock_response`) — no API key, no network.
It shows a runaway agent loop capped at `$0.50`, the ledger climbing call by
call, and the `🔌 FUSE TRIPPED` banner firing **before** the over-budget call
is ever sent.
