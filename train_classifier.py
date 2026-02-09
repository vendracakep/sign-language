import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🤖 TRAINING CLASSIFIER - IMPROVED VERSION")
print("=" * 70)

# Load data
print("\n📂 Loading data...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"✓ Dataset loaded: {len(data)} samples")
print(f"✓ Features per sample: {len(data[0])}")
print(f"✓ Unique classes: {len(set(labels))}")

# Class distribution
unique, counts = np.unique(labels, return_counts=True)
print(f"\n📊 Class distribution:")
for cls, count in zip(unique, counts):
    print(f"   Class {cls}: {count} samples ({count/len(labels)*100:.1f}%)")

# Split data
print(f"\n🔀 Splitting data (80% train, 20% test)...")
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, 
    test_size=0.2, 
    shuffle=True, 
    stratify=labels,
    random_state=42  # Reproducibility
)

print(f"✓ Train set: {len(x_train)} samples")
print(f"✓ Test set: {len(x_test)} samples")

# ============================================
# HYPERPARAMETER TUNING
# ============================================
print(f"\n🔧 Training Random Forest with optimized parameters...")

# Parameter yang sudah di-tune (bisa di-adjust)
model = RandomForestClassifier(
    n_estimators=200,        # Lebih banyak trees = lebih stabil (default: 100)
    max_depth=20,            # Cegah overfitting
    min_samples_split=5,     # Minimal sample untuk split
    min_samples_leaf=2,      # Minimal sample di leaf
    max_features='sqrt',     # Feature selection per tree
    random_state=42,
    n_jobs=-1,              # Pakai semua CPU cores
    verbose=1                # Show progress
)

model.fit(x_train, y_train)

# ============================================
# EVALUATION
# ============================================
print(f"\n📈 Evaluating model...")

# Training accuracy
y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Test accuracy
y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n{'='*70}")
print(f"📊 RESULTS:")
print(f"{'='*70}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy:     {test_accuracy * 100:.2f}%")
print(f"Overfitting Gap:   {(train_accuracy - test_accuracy) * 100:.2f}%")

if train_accuracy - test_accuracy > 0.15:
    print("⚠️  WARNING: Model might be overfitting! (gap > 15%)")
    print("   → Collect more diverse data")
    print("   → Reduce model complexity")
elif test_accuracy < 0.8:
    print("⚠️  WARNING: Low test accuracy! (< 80%)")
    print("   → Collect more data")
    print("   → Ensure better data quality")
else:
    print("✅ Model looks good!")

# Classification Report
print(f"\n📋 Detailed Classification Report:")
print("=" * 70)
labels_dict = {0: 'Halo', 1: 'Perkenalkan', 2: 'Nama Saya', 3: 'Vendra', 4: 'Terima Kasih'}
target_names = [labels_dict[int(i)] for i in unique]
print(classification_report(y_test, y_test_pred, target_names=target_names))

# Confusion Matrix
print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Cross-validation
print(f"\n🔄 Cross-validation (5-fold)...")
cv_scores = cross_val_score(model, data, labels, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# Feature importance (top 10)
print(f"\n🔝 Top 10 Most Important Features:")
feature_importance = model.feature_importances_
top_indices = np.argsort(feature_importance)[-10:][::-1]
for idx, feat_idx in enumerate(top_indices, 1):
    landmark_num = feat_idx // 2
    coord = 'x' if feat_idx % 2 == 0 else 'y'
    print(f"   {idx}. Landmark {landmark_num} ({coord}): {feature_importance[feat_idx]:.4f}")

# ============================================
# SAVE MODEL
# ============================================
print(f"\n💾 Saving model to 'model.p'...")
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("✓ Model saved!")

# Save evaluation metrics
metrics = {
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'confusion_matrix': cm,
    'classification_report': classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True)
}

with open('model_metrics.pickle', 'wb') as f:
    pickle.dump(metrics, f)
print("✓ Metrics saved to 'model_metrics.pickle'")

# ============================================
# VISUALIZATIONS
# ============================================
print(f"\n📊 Creating visualizations...")

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("✓ Confusion matrix saved to 'confusion_matrix.png'")

# 2. Feature Importance Plot
plt.figure(figsize=(12, 6))
sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("✓ Feature importance saved to 'feature_importance.png'")

# 3. Accuracy Comparison
plt.figure(figsize=(8, 6))
metrics_names = ['Train', 'Test', 'CV Mean']
metrics_values = [train_accuracy * 100, test_accuracy * 100, cv_scores.mean() * 100]
colors = ['#4CAF50', '#2196F3', '#FF9800']

bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Accuracy (%)')
plt.title('Model Performance Comparison')
plt.ylim(0, 100)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=150)
print("✓ Accuracy comparison saved to 'accuracy_comparison.png'")

plt.close('all')

# ============================================
# SUMMARY
# ============================================
print(f"\n{'='*70}")
print(f"✨ TRAINING COMPLETED!")
print(f"{'='*70}")
print(f"Model: Random Forest ({model.n_estimators} trees)")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"\n📁 Files created:")
print(f"   - model.p (trained model)")
print(f"   - model_metrics.pickle (evaluation metrics)")
print(f"   - confusion_matrix.png")
print(f"   - feature_importance.png")
print(f"   - accuracy_comparison.png")
print(f"\n🚀 Ready to use! Run: python inference.py")
print(f"{'='*70}")

# Recommendations
if test_accuracy < 0.7:
    print(f"\n⚠️  RECOMMENDATIONS TO IMPROVE:")
    print(f"   1. Collect MORE data (current: {len(data)} samples)")
    print(f"   2. Ensure VARIETY in data (different angles, distances, lighting)")
    print(f"   3. Check if gestures are DISTINCT enough")
    print(f"   4. Review confused classes in confusion matrix")
elif test_accuracy < 0.85:
    print(f"\n💡 GOOD! To improve further:")
    print(f"   1. Add more diverse data")
    print(f"   2. Fine-tune hyperparameters")
elif test_accuracy >= 0.85:
    print(f"\n🎉 EXCELLENT! Your model is performing well!")
    print(f"   Consider expanding to more gesture classes!")