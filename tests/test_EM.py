import sys
sys.path.append('src')
from utils.EvalutionMetrics import Metrics
from core.GPU import USE_GPU, cp




# Mock data
y_true = cp.array([1, 0, 1, 1, 0, 1])
y_pred = cp.array([1, 0, 1, 0, 0, 1])

y_true2 = cp.array([1, 0, 1, 0, 0, 1])
y_pred2 = cp.array([0, 0, 1, 0, 1, 1])


# Instantiate the Metrics class
metrics = Metrics(y_true, y_pred)
metrics2 = Metrics(y_true2, y_pred2)

# Test each method
print("Accuracy:", metrics.accuracy())
print("Mean Squared Error:", metrics.mean_squared_error())
print("Cross Entropy:", metrics.cross_entropy())
print("Precision Score:", metrics.precision_score())
print("Recall Score:", metrics.recall_score())
print("F1 Score:", metrics.f1_score())
# For multi_class_f1_score, you need multi-class mock data. Skipping for this example.
# metrics.multi_class_f1_score()

# Compute for batch (using the same mock data for simplicity)
metrics.compute_for_batch(y_true, y_pred)

# Visualize metrics
metrics.visualize_metrics(plot_type='line' , metrics_to_plot=["Accuracy", "Cross Entropy", "MSE"], other_metrics=metrics2  , bar_width= 0.3)

# ROC-AUC (Note: This requires sklearn and will work for binary classification mock data)
print("ROC-AUC Score:", metrics.roc_auc())