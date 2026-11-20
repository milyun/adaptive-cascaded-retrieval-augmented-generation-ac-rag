# Constant-Mode Baseline for Table 2

The modal reference is used internally for scoring and is deliberately not exported. Temperature matches within ±0.5 °C; other numeric values use the same Table 2 tolerance, and blood pressure and binary fields require exact normalized matches. Blank reference slots count as false positives; the incomplete blood-pressure reference is excluded.

| Parameter | T | Blank | Excluded | TP | FP | FN | Predicted fields | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Body Temperature (°C) | 89 | 0 | 0 | 81 | 8 | 8 | 89 | 91.0 | 91.0 | 91.0 |
| Blood Pressure (mmHg) | 87 | 1 | 1 | 4 | 84 | 83 | 88 | 4.5 | 4.6 | 4.6 |
| Pulse (bpm) | 89 | 0 | 0 | 6 | 83 | 83 | 89 | 6.7 | 6.7 | 6.7 |
| Respiratory Rate (bpm) | 82 | 7 | 0 | 50 | 39 | 32 | 89 | 56.2 | 61.0 | 58.5 |
| SpO2 (%) | 85 | 4 | 0 | 29 | 60 | 56 | 89 | 32.6 | 34.1 | 33.3 |
| Defecation | 87 | 2 | 0 | 48 | 41 | 39 | 89 | 53.9 | 55.2 | 54.5 |
| Appetite | 82 | 7 | 0 | 70 | 19 | 12 | 89 | 78.7 | 85.4 | 81.9 |
| Overall (micro) | 601 | 21 | 1 | 288 | 334 | 313 | 622 | 46.3 | 47.9 | 47.1 |

Checks: visits = 89; T = 601; predicted fields = 622; temperature TP = 81.
