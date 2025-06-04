import pandas as pd
import xgboost as xgb
import pickle

print('Creating top 20 features model for diversity and conversion rate analysis...')

# Load dataset
df = pd.read_csv('data/nesto_merge_0.csv')
print(f'Dataset shape: {df.shape}')

# Top 20 most relevant features for diversity and conversion rate analysis
top_features = [
    '2024 Visible Minority Total Population (%)',
    '2024 Visible Minority South Asian (%)',
    '2024 Visible Minority Chinese (%)',
    '2024 Visible Minority Black (%)',
    '2024 Household Average Income (Current Year $)',
    '2024 Total Population',
    '2024 Structure Type Single-Detached House (%)',
    '2024 Structure Type Apartment, Building Five or More Story (%)',
    '2024 Tenure: Owned (%)',
    '2024 Maintainers - 25 to 34 (%)',
    '2024 Maintainers - 35 to 44 (%)',
    '2024 Labour Force - Labour Employment Rate',
    '2024 Pop 15+: Married (And Not Separated) (%)',
    '2024 Household Discretionary Aggregate Income',
    '2024 Property Taxes (Shelter) (Avg)',
    '2024 Regular Mortgage Payments (Shelter) (Avg)',
    'FREQUENCY',
    'SUM_FUNDED',
    '2024-2025 Total Population % Change',
    '2024 Condominium Status - In Condo (%)'
]

# Check which features are available
available_features = [f for f in top_features if f in df.columns]
missing_features = [f for f in top_features if f not in df.columns]

print(f'Available features: {len(available_features)}')
print(f'Missing features: {len(missing_features)}')

if missing_features:
    print('Missing:', missing_features)

# Prepare data
X = df[available_features].fillna(df[available_features].median())
y = df['CONVERSION_RATE'].fillna(df['CONVERSION_RATE'].median())

print(f'Training data shape: {X.shape}')

# Train smaller, faster model
model = xgb.XGBRegressor(
    n_estimators=50,  # Small model
    max_depth=3,      # Shallow
    learning_rate=0.1,
    random_state=42
)

model.fit(X, y)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f'Model R²: {r2:.4f}')

# Save model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/feature_names.txt', 'w') as f:
    for feature in available_features:
        f.write(f'{feature}\n')

print(f'✅ Top {len(available_features)} features model created and saved')
print('Key features include:')
for i, feat in enumerate(available_features[:10]):
    print(f'  {i+1}. {feat}')
if len(available_features) > 10:
    print(f'  ... and {len(available_features) - 10} more') 