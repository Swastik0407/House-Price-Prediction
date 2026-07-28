# House Pricing Prediction

Simple project that loads a serialized model and feature pipeline to predict house prices.

## Included files

- [house_model.joblib](house_model.joblib) — trained regression model (joblib format)
- [house_features.joblib](house_features.joblib) — feature transformer / preprocessing pipeline (joblib format)
- [main.py](main.py) — example entrypoint that uses the model to produce predictions

## Requirements

- Python 3.8+
- Common packages: `numpy`, `pandas`, `scikit-learn`, `joblib`

Install dependencies (recommended inside a virtual environment):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install numpy pandas scikit-learn joblib
```

If you keep a `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

## Usage

Run the example entrypoint:

```bash
python main.py
```

Or use the model programmatically — replace the sample features with the real feature names expected by your pipeline:

```python
from joblib import load
import pandas as pd

model = load('house_model.joblib')
feature_pipeline = load('house_features.joblib')

sample = pd.DataFrame([{
    # Replace these placeholders with your actual feature names and values
    'LotArea': 8450,
    'OverallQual': 7,
    'YearBuilt': 2003,
    # ...
}])

X = feature_pipeline.transform(sample) if hasattr(feature_pipeline, 'transform') else sample
pred = model.predict(X)
print('Predicted price:', pred)
```

## Notes

- The `.joblib` files are binary artifacts that contain the trained model and preprocessing objects. Do not open them in a text editor.
- If you need to retrain the model, add a training script (not included) that saves the model and feature pipeline using `joblib.dump()`.

## Next steps

- Add a `requirements.txt` to pin dependency versions.
- Add unit tests for `main.py` and any preprocessing utilities.
- Provide a sample input CSV and a small script to validate predictions.
