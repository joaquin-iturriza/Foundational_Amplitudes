#!/bin/bash
# CLAUDE.md 0g (the user, 2026-10-25: "if you have something that needs my decision, you usually just say it in the
# message, and when I come back I answer, not prompt a choice for me to make that stops the turn until I answer"):
# the interactive question prompt ends the turn until the user replies, so it is refused here. A question goes in the
# message; only the item that needs the answer waits, everything else continues.
echo "no_blocking_question: AskUserQuestion is refused in this project (CLAUDE.md 0g). Put the question in your message, mark only that item as waiting, and continue with everything else." >&2
exit 2
