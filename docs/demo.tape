# AgentFuse demo tape — record the fuse tripping on a runaway agent.
#
# Rendered in CI via charmbracelet/vhs (see .github/workflows/demo.yml),
# or locally:
#
#     vhs docs/demo.tape          # writes assets/demo.gif
#
# Install vhs: https://github.com/charmbracelet/vhs
# The demo runs fully offline (litellm mock_response) — no API key needed.

Output assets/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1100
Set Height 640
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms

# --- intro: what AgentFuse is, and where the live ledger lives --------------
Type "# AgentFuse — a per-task spend circuit-breaker that trips BEFORE the call is sent"
Enter
Sleep 800ms

Type "agentfuse status"
Sleep 400ms
Enter
Sleep 2s

# --- the main event: a runaway agent loop, capped at $0.50 -----------------
# The ledger climbs call by call; the moment the next call WOULD cross the
# ceiling, AgentFuse prints `🔌 FUSE TRIPPED` and halts BEFORE that call is sent.
Type "# Reproduce the runaway-agent loop, offline (litellm mock), capped at $0.50"
Enter
Sleep 600ms

Type "agentfuse demo --ceiling 0.50"
Sleep 600ms
Enter

# Let the ledger climb and the fuse trip on screen.
Sleep 6s

# Hold on the 🔌 FUSE TRIPPED banner + the "spend was never incurred" summary.
Sleep 3s
