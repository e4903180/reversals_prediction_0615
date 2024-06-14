import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score

class Evaluator:
    def __init__(self, params):
        self.params = params
        pass

    def plot_reversals_ratio_multi(self, y_train_max_indices, y_val_max_indices, y_test_max_indices, y_preds_max_indices, show=True, save_path=None):
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))
        data_sets = [y_train_max_indices, y_val_max_indices, y_test_max_indices, y_preds_max_indices]
        set_labels = ['Train', 'Validation', 'Test', 'Predictions']
        reversals_labels = ['No Reversal', 'Peak', 'Valley']
        reversals_labels_color = {'No Reversal': 'green', 'Peak': 'orange', 'Valley': '#1f77b4'}
        
        for idx, y_data in enumerate(data_sets):
            y_data_label = self._change_labels(y_data, abbreviation=False)
            counts = pd.DataFrame(y_data_label).value_counts().to_dict()
            
            # 將計數字典的鍵處理為字符串
            counts = {k[0]: v for k, v in counts.items()}
            
            # 確保所有標籤都包含，即使計數為0
            count_values = [counts.get(label, 0) for label in reversals_labels]
            count_labels = reversals_labels
            
            count_colors = [reversals_labels_color[label] for label in count_labels]

            row_idx = idx // 2
            col_idx = idx % 2
            
            ax_pie = axs[row_idx, col_idx]
            patches, texts, _ = ax_pie.pie(count_values, labels=count_labels, colors=count_colors, autopct='%1.1f%%', startangle=140)
            ax_pie.axis('equal')
            ax_pie.set_title(f'Reversals Ratio ({set_labels[idx]})')
            
            labels_text = [f'{label}\n({count})' for label, count in zip(count_labels, count_values)]
            ax_pie.legend(patches, labels_text, loc='upper right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def get_confusion_matrix(self, y_test_max_indices, y_preds_max_indices, show=True, save_path=None):
        # Convert tensors to numpy arrays
        y_test_np = y_test_max_indices.numpy()
        y_preds_np = y_preds_max_indices.numpy()

        # Calculate metrics for each class
        accuracy = accuracy_score(y_test_np, y_preds_np)
        precision = precision_score(y_test_np, y_preds_np, average=None, zero_division=0, labels=[0, 1, 2])
        recall = recall_score(y_test_np, y_preds_np, average=None, zero_division=0, labels=[0, 1, 2])
        f1 = f1_score(y_test_np, y_preds_np, average=None, zero_division=0, labels=[0, 1, 2])

        # Calculate confusion matrix
        cm = confusion_matrix(y_test_np, y_preds_np, labels=[0, 1, 2])

        # Extract True Positives, True Negatives, False Positives, False Negatives for each class
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        # Calculate Specificity
        specificity = np.where((TN + FP) != 0, TN / (TN + FP), 0)

        # Calculate False Positive Rate
        fpr = np.where((FP + TN) != 0, FP / (FP + TN), 0)

        # Calculate False Negative Rate
        fnr = np.where((FN + TP) != 0, FN / (FN + TP), 0)

        # Calculate micro-averaging metrics
        micro_precision = precision_score(y_test_np, y_preds_np, average='micro', zero_division=0, labels=[0, 1, 2])
        micro_recall = recall_score(y_test_np, y_preds_np, average='micro', zero_division=0, labels=[0, 1, 2])
        micro_f1 = f1_score(y_test_np, y_preds_np, average='micro', zero_division=0, labels=[0, 1, 2])

        # Calculate macro-averaging metrics
        macro_precision = precision_score(y_test_np, y_preds_np, average='macro', zero_division=0, labels=[0, 1, 2])
        macro_recall = recall_score(y_test_np, y_preds_np, average='macro', zero_division=0, labels=[0, 1, 2])
        macro_f1 = f1_score(y_test_np, y_preds_np, average='macro', zero_division=0, labels=[0, 1, 2])

        # Create DataFrame to store metrics
        # Create DataFrame to store metrics
        unique_labels = np.unique(y_preds_np)
        labels = ['No Reversal', 'Peak', 'Valley']
        present_labels = [labels[i] for i in unique_labels]

        confusion_metrics_info = pd.DataFrame({
            'Class': labels,
            'Accuracy': [accuracy] * len(precision),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            # 'Specificity': specificity,
            # 'False Positive Rate': fpr,
            # 'False Negative Rate': fnr
        })

        # Add macro-averaging metrics
        macro_avg_info = pd.DataFrame({
            'Class': ['Macro-average'],
            'Accuracy': [accuracy],
            'Precision': [macro_precision],
            'Recall': [macro_recall],
            'F1-Score': [macro_f1],
            # 'Specificity': [specificity],
            # 'False Positive Rate': [fpr],
            # 'False Negative Rate': [fnr]
        })

        # Add micro-averaging metrics
        micro_avg_info = pd.DataFrame({
            'Class': ['Micro-average'],
            'Accuracy': [accuracy],
            'Precision': [micro_precision],
            'Recall': [micro_recall],
            'F1-Score': [micro_f1],
            # 'Specificity': [specificity],
            # 'False Positive Rate': [fpr],
            # 'False Negative Rate': [fnr]
        })

        confusion_metrics_info = pd.concat([confusion_metrics_info, macro_avg_info, micro_avg_info], ignore_index=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
        
        return confusion_metrics_info

    def get_precision_recall_curves(self, y_test, y_preds, show=True, save_path=None):
        # Convert tensors to numpy arrays
        y_test_np = y_test.numpy()
        y_preds_np = y_preds.numpy()

        # Compute precision and recall for each class
        precision = dict()
        recall = dict()
        for i in range(y_test_np.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(y_test_np[:, i], y_preds_np[:, i])
        pr_auc = list(range(y_test_np.shape[1]))
        labels = ['No Reversal', 'Peak', 'Valley']
        # Plot Precision-Recall curves for each class
        plt.figure(figsize=(10, 6))
        for i in range(y_test_np.shape[1]):
            pr_auc[i] = -round(np.trapz(precision[i], recall[i]), 4)
            plt.plot(recall[i], precision[i], lw=2, label='{}, auc={:.4f}'.format(labels[i], pr_auc[i]))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
        return pr_auc
            
    def get_roc_curves(self, y_test, y_preds, show=True, save_path=None):
        # Convert tensors to numpy arrays
        y_test_np = y_test.numpy()
        y_preds_np = y_preds.numpy()

        # Compute ROC curve and ROC area for each class
        plt.figure(figsize=(10, 6))
        roc_auc = list(range(y_test_np.shape[1]))
        labels = ['No Reversal', 'Peak', 'Valley']
        for i in range(y_test_np.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_np[:, i], y_preds_np[:, i])
            roc_auc[i] = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label='{} (AUC = {:.4f})'.format(labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

        return roc_auc

    def plot_training_curve(self, history, show=True,save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss and validation loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        rollback_epoch = history.get('rollback_epoch')
        if rollback_epoch is not None:
            ax1.axvline(x=rollback_epoch, color='r', linestyle='--', label='Rollback Epoch')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()

        # Plot accuracy and validation accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        if rollback_epoch is not None:
            ax2.axvline(x=rollback_epoch, color='r', linestyle='--', label='Rollback Epoch')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True)
        ax2.legend()
        
        fig.suptitle(f'Training Curve, rollback_epoch: {rollback_epoch}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_val_training_curve(self, val_train_history, show=True,save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss and validation loss
        ax1.plot(val_train_history['epoch_loss'][0], label='Val Training Loss')
        ax1.set_title('Val Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()

        # Plot accuracy and validation accuracy
        ax2.plot(val_train_history['epoch_accuracy'][0], label='Val Training Accuracy')

        ax2.set_title('Val Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True)
        ax2.legend()

        fig.suptitle(f'Val Training Curve')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_online_training_curve(self, online_history, show=True,save_path=None):
        # Create a figure with two subplots
        fig, ax = plt.subplots(2, 2, figsize=(20, 8))
        fig.suptitle('Online Training Curve')
        # Plot loss on the second subplot
        ax[0, 0].plot(online_history['loss'], label='Training Loss')
        ax[0, 0].set_title('Online training loss for the first epoch in each training session')
        ax[0, 0].set_xlabel('Time')
        ax[0, 0].set_ylabel('Loss')
        ax[0, 0].legend()
        ax[0, 0].grid(True)

        # Plot accuracy on the first subplot
        ax[0, 1].plot(online_history['accuracy'], color='tab:red', label='Training Accuracy')
        ax[0, 1].set_title('Online Training Accuracy for the first epoch in each training session')
        ax[0, 1].set_xlabel('Time')
        ax[0, 1].set_ylabel('Accuracy')
        ax[0, 1].set_ylim([0, 1])
        ax[0, 1].legend()
        ax[0, 1].grid(True)

        # ax[1, 0].plot(np.array(online_history['epoch_loss']).flatten(), color='tab:blue', label='Training Loss')
        for epoch in range(0, len(online_history['epoch_loss'])):
            ax[1, 0].plot(range(epoch*len(online_history['epoch_loss'][epoch]), (epoch+1)*len(online_history['epoch_loss'][epoch])), online_history['epoch_loss'][epoch], label='Training Loss' if epoch == 0 else '')
        ax[1, 0].set_title('Online Training Loss for each epoch in each training session')
        ax[1, 0].set_xlabel('epoch')
        ax[1, 0].set_ylabel('Loss')
        ax[1, 0].legend()
        ax[1, 0].grid(True)

        # Plot accuracy on the first subplot
        for epoch in range(0, len(online_history['epoch_accuracy'])):
            ax[1, 1].plot(range(epoch*len(online_history['epoch_accuracy'][epoch]), (epoch+1)*len(online_history['epoch_accuracy'][epoch])), online_history['epoch_accuracy'][epoch], label='Training Accuracy' if epoch == 0 else '')
        ax[1, 1].set_title('Online Training Accuracy for each epoch in each training session')
        ax[1, 1].set_xlabel('epoch')
        ax[1, 1].set_ylabel('Accuracy')
        ax[1, 1].set_ylim([0, 1])
        ax[1, 1].legend()
        ax[1, 1].grid(True)
        # Adjust the layout
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def _kbar(self, open, close, high, low, pos, ax):  # for US stocks
        if close > open:
            color = 'green'   # rise
            height = close - open
            bottom = open
        else:
            color = 'red'     # fall
            height = open - close
            bottom = close
        ax.bar(pos, height=height, bottom=bottom, width=0.6, color=color)
        ax.vlines(pos, high, low, color=color)
        
    def _change_labels(self, signal, abbreviation=False):
        signal_np = signal.numpy()
        if abbreviation == True:
            signal_np = np.where(signal_np.astype(str) == '0', "N", signal_np)
            signal_np = np.where(signal_np.astype(str) == '1', "P", signal_np)
            signal_np = np.where(signal_np.astype(str) == '2', "V", signal_np)
        elif abbreviation == False:
            signal_np = np.where(signal_np.astype(str) == '0', "No Reversal", signal_np)
            signal_np = np.where(signal_np.astype(str) == '1', "Peak", signal_np)
            signal_np = np.where(signal_np.astype(str) == '2', "Valley", signal_np)
        return signal_np

    def plot_stock_and_predictions(self, y_test_max_indices, y_preds_max_indices, test_dataset, test_dates, show=True, save_path=None):
        fig, ax = plt.subplots(2, 1, figsize=(20, 12), sharex=True, height_ratios=[3, 1])

        for idx in test_dataset.loc[test_dates[0][0]: test_dates[-1][-1]].index:
            self._kbar(test_dataset.loc[idx]['Open'], test_dataset.loc[idx]['Close'], test_dataset.loc[idx]['High'], test_dataset.loc[idx]['Low'], idx, ax[0])
        ax[0].plot(test_dataset['MA'].loc[test_dates[0][0]: test_dates[-1][-1]], label='MA', color='red', linestyle='--')

        peaks_labeled = False
        valleys_labeled = False
        reversal_type = self.params["features_params"][0]["local_type"]
        for idx in test_dataset.loc[test_dates[0][0]: test_dates[-1][-1]].index:
            if test_dataset['Reversals'].loc[idx] == 1:
                ax[0].scatter(idx, test_dataset.loc[idx][reversal_type], color='darkgreen', label='Peaks' if peaks_labeled == False else '', zorder=5, marker='v', s=100)
                peaks_labeled = True
            elif test_dataset['Reversals'].loc[idx] == 2:
                ax[0].scatter(idx, test_dataset.loc[idx][reversal_type], color='darkorange', label='Valleys' if valleys_labeled == False else '', zorder=5, marker='^', s=100)
                valleys_labeled = True
        ax[0].set_title('Stock Price')
        ax[0].grid(True)
        ax[0].legend()

        
        y_test_label = self._change_labels(y_test_max_indices, abbreviation=True)
        y_preds_label = self._change_labels(y_preds_max_indices, abbreviation=True)
        
        for idx in range(test_dates.shape[0]):
            ax[1].plot(test_dataset['Reversals'].loc[test_dates[idx]])
            ax[0].axvline(x=test_dates[idx][0], color='lightgray', linestyle='--')
            ax[1].axvline(x=test_dates[idx][0], color='lightgray', linestyle='--')
            ax[1].annotate(f"{y_test_label[idx]}", (test_dates[idx][0], 1.7), label='actual')
            ax[1].annotate(f"{y_preds_label[idx]}", (test_dates[idx][0], 1.9), color='red', label='predicted')
        ax[1].set_title('Reversals')
        ax[1].set_ylabel('Class')
        ax[1].set_yticks([0, 1, 2])
        ax[1].grid(True)
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

            
    def get_results(self, y_train, y_val, y_test, y_preds, test_dataset, test_dates, history, online_history, show=True):
        y_train_max_indices = np.argmax(y_train, axis=-1)
        y_val_max_indices = np.argmax(y_val, axis=-1)
        y_test_max_indices = np.argmax(y_test, axis=-1)
        y_preds_max_indices = np.argmax(y_preds, axis=-1)
        
        confusion_metrics_info = self.get_confusion_matrix(y_test_max_indices, y_preds_max_indices, 
                                            show=show, save_path=self.params.get('confusion_matrix_path'))
        self.plot_reversals_ratio_multi(y_train_max_indices, y_val_max_indices, y_test_max_indices, 
                        y_preds_max_indices, show=show, 
                        save_path=self.params.get('reversals_ratio_path'))
        kappa = cohen_kappa_score(y_test_max_indices, y_preds_max_indices)
        mcc = matthews_corrcoef(y_test_max_indices, y_preds_max_indices)
        pr_auc = self.get_precision_recall_curves(y_test, y_preds, show=show, 
                                                  save_path=self.params.get('pr_auc_path'))
        rou_auc = self.get_roc_curves(y_test, y_preds, show=show, save_path=self.params.get('roc_auc_path'))
        self.plot_training_curve(history, show=show, save_path=self.params.get('training_curve_path'))
        self.plot_online_training_curve(online_history, show=show, save_path=self.params.get('online_training_curve_path'))
        self.plot_stock_and_predictions(y_test_max_indices, y_preds_max_indices, test_dataset, test_dates,
                                        show=show, save_path=self.params.get('stock_and_predictions_path'))
        
        results = {
            "confusion metrics": confusion_metrics_info.to_dict(),
            "pr_auc": pr_auc,
            "roc_auc": rou_auc,
            "kappa": kappa,
            "mcc": mcc
            }
        
        return results