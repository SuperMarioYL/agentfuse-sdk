# AgentFuse demo tape — record the fuse tripping on a runaway agent.
#
# Regenerate the GIF (not committed) with charmbracelet/vhs:
#
#     vhs assets/demo.tape          # writes assets/demo.gif
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
Set TypingSpeed 60ms

# --- intro: install + a peek at status -------------------------------------
Type "pip install agentfuse"
Sleep 600ms
Enter
Sleep 1s

Type "agentfuse status"
Sleep 400ms
Enter
Sleep 2s

# --- the main event: a runaway agent loop, capped at $0.50 -----------------
# The ledger climbs call by call; the moment the next call would cross the
# ceiling, AgentFuse prints `🔌 FUSE TRIPPED` and halts BEFORE that call is sent.
Type "agentfuse demo --ceiling 0.50"
Sleep 600ms
Enter

# Let the ledger climb and the fuse trip on screen.
Sleep 6s

# Hold on the 🔌 FUSE TRIPPED banner + the "spend was never incurred" summary.
Sleep 3s
