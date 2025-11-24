# All required libraries are imported here for you.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("C:/Users/Dell/Downloads/soil_measures.csv")

# Split into feature and target sets
X = crops.drop(columns="crop")
y = crops["crop"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create dictionaries to store models and performance
feature_performance = {}
feature_models = {}

# Train a logistic regression model for each feature
for feature in X.columns:
    log_reg = LogisticRegression(max_iter=5000, random_state=42)
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    
    # Calculate F1 score
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = f1
    feature_models[feature] = log_reg
    
    print(f"✅ {feature} model trained - F1: {f1}")

# Find the best feature
best_feature = max(feature_performance, key=feature_performance.get)
best_f1 = feature_performance[best_feature]

print(f"\n BEST FEATURE: {best_feature} (F1-score: {best_f1})")

# ============================================================================
# EXTRACT REAL PATTERNS WITHOUT HARDCODING
# ============================================================================

print("PLANTING GUIDE")
def extract_real_patterns(feature, model, feature_data):
    """Extract what the model actually learned without hardcoding"""
    
    # Use the actual data range from training
    min_val = feature_data.min()
    max_val = feature_data.max()
    
    # Test across the entire range to find natural breakpoints
    test_values = np.linspace(min_val, max_val, 100)
    predictions = model.predict(test_values.reshape(-1, 1))
    
    # Find where predictions change (decision boundaries)
    change_points = []
    current_pred = predictions[0]
    
    for i, (val, pred) in enumerate(zip(test_values, predictions)):
        if pred != current_pred:
            change_points.append((val, current_pred, pred))
            current_pred = pred
    
    # Group values by predicted crop
    crop_ranges = {}
    for i, (val, pred) in enumerate(zip(test_values, predictions)):
        if pred not in crop_ranges:
            crop_ranges[pred] = []
        crop_ranges[pred].append(val)
    
    # Calculate typical range for each crop
    crop_summary = {}
    for crop, values in crop_ranges.items():
        if values:  # Check if list is not empty
            crop_summary[crop] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'count': len(values)
            }
    
    return crop_summary, change_points

# Analyze the best feature model
best_model = feature_models[best_feature]
crop_patterns, decision_points = extract_real_patterns(best_feature, best_model, X_train[best_feature])

print(f"\n🌿 WHAT THE {best_feature.upper()} MODEL ACTUALLY LEARNED:")
print(f"Based on {len(X_train)} training samples")

# Show crops in order of their typical values
sorted_crops = sorted(crop_patterns.items(), key=lambda x: x[1]['mean'])

print(f"\n📊 {best_feature.upper()} RANGES FOR EACH CROP:")
for crop, stats in sorted_crops:
    print(f"   • {crop:15}: {stats['min']:5.1f} - {stats['max']:5.1f} (avg: {stats['mean']:5.1f})")

# Show decision boundaries
print(f"\n🎯 DECISION BOUNDARIES:")
for i, (boundary, from_crop, to_crop) in enumerate(decision_points[:5]):  # Show first 5
    print(f"   • At {boundary:5.1f}: {from_crop} → {to_crop}")

# Generate automatic planting guide using percentiles
print(f"\n🌱 AUTOMATIC PLANTING GUIDE FOR {best_feature.upper()}:")
percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]
levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

for level, pctl in zip(levels, percentiles):
    value = X_train[best_feature].quantile(pctl)
    prediction = best_model.predict([[value]])[0]
    confidence = max(best_model.predict_proba([[value]])[0])
    
    print(f"   • {level:9} ({value:5.1f}) → {prediction} (confidence: {confidence:.2f})")

# Show what happens at extreme values
print(f"\n⚠️  EXTREME VALUES:")
extremes = [
    ('Minimum', X_train[best_feature].min()),
    ('Maximum', X_train[best_feature].max())
]

for desc, value in extremes:
    prediction = best_model.predict([[value]])[0]
    confidence = max(best_model.predict_proba([[value]])[0])
    print(f"   • {desc:8} ({value:5.1f}) → {prediction} (confidence: {confidence:.2f})")

# ============================================================================
# PRACTICAL RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("FARMER'S ACTION PLAN")
print("="*70)

print(f"""
1. MEASURE {best_feature.upper()} FIRST
   • Test the {best_feature} level in your soil
   • This single measurement gives you {best_f1:.1%} accuracy

2. USE THIS GUIDE:
   • Measure your soil's {best_feature} level
   • Find where it falls in the ranges above  
   • Plant the recommended crop for that range

3. EXAMPLE:
   • If your {best_feature} = 45.0, plant: {best_model.predict([[45.0]])[0]}
   • If your {best_feature} = 75.0, plant: {best_model.predict([[75.0]])[0]}

4. BUDGET SAVING:
   • Instead of testing all 4 parameters (expensive)
   • Just test {best_feature} and use this guide
""")

best_predictive_feature = {best_feature: best_f1}
print(f"📋 FINAL RESULT: {best_predictive_feature}")
