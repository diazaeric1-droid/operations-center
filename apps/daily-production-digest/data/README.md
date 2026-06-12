# Data

All data is **synthetic**. No proprietary operator data is included or should be committed.

`synthetic/generate_fleet.py` produces:
- 50 wells × 30 days of daily SCADA
- 4 wells with seeded HIGH-severity anomalies (rate drop, intake collapse, motor temp spike, runtime degradation)
- 2 wells with seeded MEDIUM amps creep
- Remaining 44 wells healthy

```
python data/synthetic/generate_fleet.py
```

Run before `python -m src.scheduler`.
