from pathlib import Path
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

import seaborn as sns
sns.set_theme(style="whitegrid", context="notebook")
sns.set_palette(palette="muted")
plt.rcParams["figure.dpi"] = 500
plt.rcParams["savefig.dpi"] = 800
plt.rcParams["font.size"] = 12
plt.rcParams["lines.linewidth"] = 1.5


DEFAULT_LITHOLOGY_KEYS: Dict[int, str] = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}


class LithologyEvaluator:
    """Класс для визуализации и оценки качества моделей предсказания литологии."""

    def __init__(self, mapping: Optional[Dict[int, str]] = None):
        self.mapping = mapping or DEFAULT_LITHOLOGY_KEYS

    def plot_confusion_matrix(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Строит нормированную матрицу ошибок."""
        labels_sorted = sorted(np.unique(y_true))
        labels_names = [self.mapping.get(l, str(l)) for l in labels_sorted]

        cm = confusion_matrix(
            y_true, y_pred, labels=labels_sorted, normalize="true"
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels_names
        )
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f")
        ax.set_title("Confusion Matrix (normalized by true label)")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Матрица ошибок сохранена в: {save_path}")

        plt.show()

    def plot_roc_auc(
        self,
        y_true: Union[pd.Series, np.ndarray],
        val_probs: np.ndarray,
        classes: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Строит кривые ROC-AUC для каждого класса (One-vs-Rest)."""
        sns.set_theme(style="whitegrid", context="notebook")
        sns.set_palette(palette="muted")

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        for i, class_code in enumerate(classes):
            y_true_binary = (y_true == class_code).astype(int)

            if y_true_binary.sum() == 0:
                continue

            y_score = val_probs[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)

            class_name = self.mapping.get(class_code, str(class_code))
            ax.plot(
                fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})"
            )

        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            linestyle="--",
            lw=1.5,
            label="Random (AUC = 0.50)",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
        ax.set_title(
            "ROC-AUC Curves per Lithology Class (One-vs-Rest)", fontsize=13
        )
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"ROC-AUC графики сохранены в: {save_path}")

        plt.show()

    def plot_all(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        val_probs: np.ndarray,
        classes: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Позволяет построить и сохранить сразу все графики метрик."""
        cm_path = (
            Path(output_dir) / "confusion_matrix.png" if output_dir else None
        )
        roc_path = (
            Path(output_dir) / "roc_auc_curves.png" if output_dir else None
        )

        self.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        self.plot_roc_auc(y_true, val_probs, classes, save_path=roc_path)