### Coverage and detection power, window by window

| suite | metric | w0 | w1 | w2 | w3 |
|---|---|---|---|---|---|
| `frozen` | coverage | 0.71 | 0.48 | 0.38 | 0.35 |
| `frozen` | power (averaged over clusters) | 0.43 | 0.46 | 0.36 | 0.49 |
| `frozen` | power in its **worst** cluster | 0.250 | 0.117 | 0.000 | 0.000 |
| `mined` | coverage | 0.71 | 0.51 | 0.50 | 0.56 |
| `mined` | power (averaged over clusters) | 0.42 | 0.48 | 0.46 | 0.48 |
| `mined` | power in its **worst** cluster | 0.300 | 0.100 | 0.017 | 0.067 |

### Frozen suite against the final window

| suite | cases | traffic | coverage | 95% CI | radius | q25 | q75 |
|---|---|---|---|---|---|---|---|
| `frozen` | 120 | 225 | 0.347 | [0.288, 0.411] | 0.240 | 0.209 | 0.667 |

### False-alarm rate under no regression

| suite size | threshold | threshold gate fires | paired gate fires |
|---|---|---|---|
| 30 | 0.900 | **0.178** [0.129, 0.240] | 0.000 [0.000, 0.021] |
| 100 | 0.849 | **0.128** [0.087, 0.184] | 0.000 [0.000, 0.021] |

### Judge depth: designed against measured

| scenario | designed depth | measured depth | majority | length | keyword | bow | charngram |
|---|---|---|---|---|---|---|---|
| `support_triage` | keyword | **keyword** | 0.000 | 0.125 | 0.879 | 0.879 | 0.873 |
| `rag_grounding` | bow | **bow** | 0.000 | 0.225 | 0.306 | 0.861 | 0.861 |
| `code_patch` | charngram | **charngram** | 0.000 | 0.295 | -0.026 | -0.033 | 0.887 |
| `tool_calling` | none | **none** | 0.000 | 0.076 | -0.020 | 0.097 | 0.104 |
