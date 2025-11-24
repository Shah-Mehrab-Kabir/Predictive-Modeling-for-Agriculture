🌱 TRAINING MODELS AND EXTRACTING REAL PATTERNS
======================================================================
✅ N model trained - F1: 0.1041
✅ P model trained - F1: 0.1282
✅ K model trained - F1: 0.1897
✅ ph model trained - F1: 0.0458

🎯 BEST FEATURE: K (F1-score: 0.1897)

======================================================================
REAL PLANTING GUIDE (From Model Patterns)
======================================================================

🌿 WHAT THE K MODEL ACTUALLY LEARNED:
Based on 1760 training samples

📊 K RANGES FOR EACH CROP:
   • orange         :   5.0 -  13.1 (avg:   9.0)
   • lentil         :  15.1 -  19.1 (avg:  17.1)
   • kidneybeans    :  21.2 -  21.2 (avg:  21.2)
   • pigeonpeas     :  23.2 -  23.2 (avg:  23.2)
   • coffee         :  25.2 -  31.3 (avg:  28.2)
   • coconut        :  33.3 -  33.3 (avg:  33.3)
   • rice           :  35.3 -  41.4 (avg:  38.3)
   • pomegranate    :  43.4 -  43.4 (avg:  43.4)
   • watermelon     :  45.4 -  47.4 (avg:  46.4)
   • muskmelon      :  49.4 -  65.6 (avg:  57.5)
   • chickpea       :  67.6 - 188.8 (avg: 128.2)
   • apple          : 190.9 - 194.9 (avg: 192.9)
   • grapes         : 196.9 - 205.0 (avg: 201.0)

🎯 DECISION BOUNDARIES:
   • At  15.1: orange → lentil
   • At  21.2: lentil → kidneybeans
   • At  23.2: kidneybeans → pigeonpeas
   • At  25.2: pigeonpeas → coffee
   • At  33.3: coffee → coconut

🌱 AUTOMATIC PLANTING GUIDE FOR K:
   • Very Low  ( 16.0) → lentil (confidence: 0.16)
   • Low       ( 22.0) → pigeonpeas (confidence: 0.13)
   • Medium    ( 31.0) → coffee (confidence: 0.34)
   • High      ( 45.0) → watermelon (confidence: 0.16)
   • Very High ( 84.0) → chickpea (confidence: 1.00)

⚠️  EXTREME VALUES:
   • Minimum  (  5.0) → orange (confidence: 1.00)
   • Maximum  (205.0) → grapes (confidence: 0.57)

======================================================================
FARMER'S ACTION PLAN
======================================================================

1. MEASURE K FIRST
   • Test the K level in your soil
   • This single measurement gives you 19.0% accuracy

2. USE THIS GUIDE:
   • Measure your soil's K level
   • Find where it falls in the ranges above  
   • Plant the recommended crop for that range

3. EXAMPLE:
   • If your K = 45.0, plant: watermelon
   • If your K = 75.0, plant: chickpea

4. BUDGET SAVING:
   • Instead of testing all 4 parameters (expensive)
   • Just test K and use this guide

📋 FINAL RESULT: {'K': 0.1897181811666794}
