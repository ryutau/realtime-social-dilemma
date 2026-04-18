# Source Code

## Structure

```
src/
├── data/               # Data loading & preprocessing package
│   ├── __init__.py      # Public API: load_pilot, load_main, preprocess_*
│   ├── loader.py        # Preprocessing and loading logic
│   └── utils.py         # Project path constants
└── (analysis scripts)   # One script per figure / regression / table
```

## Usage

```python
from src.data import load_pilot, load_main

pilot_df = load_pilot()
main_df = load_main()
```

## Analysis script naming convention

- `01_*.py` - Descriptive statistics and demographic summaries
- `02_*.py` - Main analyses
- `03_*.py` - Figures and visualizations
