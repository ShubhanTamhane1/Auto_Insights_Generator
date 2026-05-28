# auto-insights

Automated EDA for pandas DataFrames, powered by the Anthropic API.

## Installation
pip install auto-insights

## Quickstart
```python
from auto_insights import InsightsGenerator
report = InsightsGenerator(df).run()
report.save("report.html")
```

## Requirements
Set ANTHROPIC_API_KEY in your environment.
