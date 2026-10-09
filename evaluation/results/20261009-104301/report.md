# Robot tool-use benchmark

Model `gpt-6-luna`, all MCP tools offered every step.

- success: **6/7 (86%)**
- LLM claimed success but the check failed: 1
- avg steps 2.7, avg time 15s (LLM 14s), avg tokens in 57,862 / out 306

| task | kind | success | avg steps | avg time (s) | avg tokens in |
|---|---|---|---|---|---|
| peace_sign | gesture | 1/1 | 1.0 | 8 | 30,908 |
| point | gesture | 1/1 | 1.0 | 10 | 30,897 |
| fist | gesture | 1/1 | 2.0 | 13 | 46,351 |
| hover_ball | positioning | 1/1 | 1.0 | 8 | 30,910 |
| palm_left | positioning | 1/1 | 1.0 | 5 | 30,896 |
| bend_index_only | gesture | 0/1 | 1.0 | 10 | 30,914 |
| ball_to_zone | manipulation | 1/1 | 12.0 | 51 | 204,160 |
