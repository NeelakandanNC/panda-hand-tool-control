# Robot tool-use benchmark

Model `gpt-6-luna`, all MCP tools offered every step.

- success: **3/3 (100%)**
- LLM claimed success but the check failed: 0
- avg steps 6.3, avg time 25s (LLM 23s), avg tokens in 114,350 / out 436

| task | kind | success | avg steps | avg time (s) | avg tokens in |
|---|---|---|---|---|---|
| lift_cube | manipulation | 1/1 | 4.0 | 18 | 77,654 |
| cylinder_to_zone | manipulation | 1/1 | 7.0 | 27 | 124,857 |
| cube_to_zone | manipulation | 1/1 | 8.0 | 29 | 140,539 |
