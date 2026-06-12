# assets/

Demo media for AgentFuse.

| file | what it is | committed? |
| --- | --- | --- |
| `demo.tape` | [charmbracelet/vhs](https://github.com/charmbracelet/vhs) script that records the demo | yes |
| `demo.gif` | the recorded terminal GIF embedded in the README | no — generated |

## Regenerate the GIF

```bash
# install vhs once: https://github.com/charmbracelet/vhs
vhs assets/demo.tape   # writes assets/demo.gif
```

The demo runs fully offline (litellm `mock_response`) — no API key, no network.
It shows a runaway agent loop capped at `$0.50`, the ledger climbing call by
call, and the `🔌 FUSE TRIPPED` banner firing **before** the over-budget call
is ever sent.
