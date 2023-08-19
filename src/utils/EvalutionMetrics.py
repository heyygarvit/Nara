
import sys
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

sys.path.append('src')
from core.GPU import USE_GPU, cp
from core.Tensor import Tensor

logging.basicConfig(level=logging.INFO)


class Metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self._check_shapes()

    def _check_shapes(self):
        """Check if y_true and y_pred have the same shape."""
        if self.y_true.shape != self.y_pred.shape:
            logging.error("Shape mismatch between y_true and y_pred.")
            raise ValueError("y_true and y_pred must have the same shape.")

    def accuracy(self, weights=None):
        """Compute the accuracy of predictions."""
        correct = cp.sum(self.y_true == self.y_pred)
        total = self.y_true.size
        if weights is not None:
            return cp.sum(weights * (self.y_true == self.y_pred)) / cp.sum(weights)
        return correct / total

    # def mean_squared_error(self):
    #     """Compute the mean squared error."""
    #     return cp.mean((self.y_true - self.y_pred) ** 2)
    
    def mean_squared_error(y_true, y_pred):
        """Compute the mean squared error."""
        y_true = Tensor._ensure_tensor(y_true)
        y_pred = Tensor._ensure_tensor(y_pred)
        
        mse_value = (y_true - y_pred) ** 2
        return mse_value.mean()

    def cross_entropy(self):
        """Compute the cross-entropy loss."""
        epsilon = 1e-15  # To avoid log(0)
        y_pred_clipped = cp.clip(self.y_pred, epsilon, 1 - epsilon)
        return -cp.mean(self.y_true * cp.log(y_pred_clipped))

    def precision_score(self):
        """Compute the precision score."""
        true_positives = cp.sum(self.y_true * self.y_pred)
        predicted_positives = cp.sum(self.y_pred)
        return true_positives / (predicted_positives + 1e-15)

    def recall_score(self):
        """Compute the recall score."""
        true_positives = cp.sum(self.y_true * self.y_pred)
        actual_positives = cp.sum(self.y_true)
        return true_positives / (actual_positives + 1e-15)

    def f1_score(self, y_true=None, y_pred=None):
        """Compute the F1 score."""
        if y_true is None or y_pred is None:
            y_true = self.y_true
            y_pred = self.y_pred
        true_positives = cp.sum(y_true * y_pred)
        predicted_positives = cp.sum(y_pred)
        actual_positives = cp.sum(y_true)
        precision = true_positives / (predicted_positives + 1e-15)
        recall = true_positives / (actual_positives + 1e-15)
        return 2 * (precision * recall) / (precision + recall + 1e-15)

    def multi_class_f1_score(self, average='macro'):
        """Compute the F1 score for multi-class classification."""
        if len(self.y_true.shape) == 1:  # Binary classification
            return self.f1_score()
        
        num_classes = self.y_true.shape[1]
        f1_scores = []
        for i in range(num_classes):
            f1 = self.f1_score(self.y_true[:, i], self.y_pred[:, i])
            f1_scores.append(f1)
        if average == 'macro':
            return cp.mean(f1_scores)
        elif average == 'micro':
            total_true_positives = cp.sum([self.y_true[:, i] * self.y_pred[:, i] for i in range(num_classes)])
            total_predicted_positives = cp.sum(self.y_pred)
            total_actual_positives = cp.sum(self.y_true)
            precision = total_true_positives / total_predicted_positives
            recall = total_true_positives / total_actual_positives
            return 2 * (precision * recall) / (precision + recall + 1e-15)
        else:
            raise ValueError("Invalid average type. Choose either 'macro' or 'micro'.")


    def compute_for_batch(self, y_true_batch, y_pred_batch):
        """Compute metrics for a batch of data."""
        self.y_true = y_true_batch
        self.y_pred = y_pred_batch
        self._check_shapes()
        logging.info(f"Accuracy: {self.accuracy()}")
        logging.info(f"Mean Squared Error: {self.mean_squared_error()}")
        logging.info(f"F1 Score: {self.f1_score()}")

    def visualize_metrics(self, metrics_to_plot=None, other_metrics=None, bar_width=0.35, plot_type='bar', title='Metrics Visualization', ylabel='Score', xlabel='Metrics', plot_roc=False, dataset1_label='Dataset 1', dataset2_label='Dataset 2', **kwargs):
        """
        Visualize metrics using matplotlib.
        
        Parameters:
        - metrics_to_plot: List of metrics to plot. Default is ['Accuracy', 'MSE', 'F1 Score'].
        - plot_type: Type of plot. Default is 'bar'.
        - title: Title of the plot.
        - ylabel: Y-axis label.
        - xlabel: X-axis label.
        - plot_roc: Whether to plot the ROC curve. Default is False.
        - **kwargs: Other keyword arguments for the plot.
        """
        
        # All available metrics
        all_metrics = {
            "Accuracy": self.accuracy(),
            "MSE": self.mean_squared_error(),
            "F1 Score": self.f1_score(),
            "Precision Score" : self.precision_score(),
            "Recall" : self.recall_score(),
            "Cross Entropy" : self.cross_entropy(),
            "Multi Class F1 Score" : self.multi_class_f1_score()
            # Add other metrics here if needed
        }

        # If another Metrics object is provided, get its metrics
        if other_metrics:
            other_all_metrics = {
                
                "Accuracy": other_metrics.accuracy(),
                "MSE": other_metrics.mean_squared_error(),
                "F1 Score": other_metrics.f1_score(),
                "Precision Score" : other_metrics.precision_score(),
                "Recall" : other_metrics.recall_score(),
                "Cross Entropy" : other_metrics.cross_entropy(),
                "Multi Class F1 Score" : other_metrics.multi_class_f1_score()
            }
        else:
            other_all_metrics = {key: None for key in all_metrics.keys()}
        
        # If no specific metrics are provided, plot all
        if metrics_to_plot is None:
            metrics_to_plot = all_metrics.keys()
        
        # Filter out the metrics that are not in the list to be plotted
        metrics = {key: all_metrics[key] for key in metrics_to_plot}
        other_values = [other_all_metrics[key] for key in metrics_to_plot]
        
        names = list(metrics.keys())
        values = list(metrics.values())

     
        if plot_type == 'bar':
            ind = cp.arange(len(names))
            if other_metrics:
                bars1 = plt.bar(ind - bar_width/2, values, bar_width, label=dataset1_label, **kwargs)
                bars2 = plt.bar(ind + bar_width/2, other_values, bar_width, label=dataset2_label, **kwargs)
                plt.xticks(ind, names)  # Set the x-ticks to the correct names
                plt.legend()
            else:
                plt.bar(ind, values, bar_width, **kwargs)
        elif plot_type == 'line':
            line1, = plt.plot(names, values, marker='o', label=dataset1_label, **kwargs)
            if other_metrics:
                line2, = plt.plot(names, other_values, marker='x', linestyle='--', label=dataset2_label, **kwargs)
            plt.legend()  # This ensures that the legend is displayed


        # Add other plot types if needed
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        
        if plot_roc:
            if USE_GPU:
                y_true_cpu = cp.asnumpy(self.y_true)
                y_pred_cpu = cp.asnumpy(self.y_pred)
            else:
                y_true_cpu = self.y_true
                y_pred_cpu = self.y_pred
            # y_true_cpu = cp.asnumpy(self.y_true)
            # y_pred_cpu = cp.asnumpy(self.y_pred)
            fpr, tpr, _ = roc_curve(y_true_cpu, y_pred_cpu)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {self.roc_auc():.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='best')
        
        plt.show()

    def roc_auc(self):
        """Compute the ROC-AUC score."""
        if USE_GPU:
            y_true_cpu = cp.asnumpy(self.y_true)
            y_pred_cpu = cp.asnumpy(self.y_pred)
        else:
            y_true_cpu = self.y_true
            y_pred_cpu = self.y_pred

        if len(y_true_cpu.shape) == 1 or y_true_cpu.shape[1] == 1:
            # Binary classification
            return roc_auc_score(y_true_cpu, y_pred_cpu)
        else:
            # Multi-class classification
            scores = []
            for i in range(y_true_cpu.shape[1]):
                score = roc_auc_score(y_true_cpu[:, i], y_pred_cpu[:, i])
                scores.append(score)
            return cp.mean(scores)

