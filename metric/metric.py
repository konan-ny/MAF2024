import sys
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.manifold import TSNE

from aif360.metrics import DatasetMetric
from MAF2024.datamodule.dataset import aifData


class DataMetric:
    def __init__(self, dataset):
        self.dataset = dataset
        self.label_names = dataset.label_names
        self.favorable_label = dataset.favorable_label
        self.unfavorable_label = dataset.unfavorable_label
        self.protected_attribute_names = dataset.protected_attribute_names
        self.privileged_protected_attributes = dataset.privileged_protected_attributes
        self.unprivileged_protected_attributes = (
            dataset.unprivileged_protected_attributes
        )
        self.df = dataset.convert_to_dataframe()[0]

    def select_data_by_privilege(self, selected_data: pd.DataFrame, privileged=None):
        p_names = self.protected_attribute_names
        if privileged == True:
            for idx, pn in enumerate(p_names):
                selected_data = selected_data[
                    selected_data[pn] == self.privileged_protected_attributes[idx][0]
                ].copy()
        elif privileged == False:
            for idx, pn in enumerate(p_names):
                selected_data = selected_data[
                    selected_data[pn] == self.unprivileged_protected_attributes[idx][0]
                ].copy()
        return selected_data

    def num_positive(self, privileged: bool = False):
        df = self.df.copy()
        for ln in self.label_names:
            df = df[df[ln] == self.favorable_label].copy()
        return len(
            self.select_data_by_privilege(selected_data=df, privileged=privileged)
        )

    def num_negative(self, privileged: bool = False):
        df = self.df.copy()
        for ln in self.label_names:
            df = df[df[ln] == self.unfavorable_label].copy()
        return len(
            self.select_data_by_privilege(selected_data=df, privileged=privileged)
        )

    def base_rate(self, privileged: bool = False):
        df = self.df.copy()
        p = self.num_positive(privileged=privileged)
        n = len(self.select_data_by_privilege(selected_data=df, privileged=privileged))
        return p / n

    def disparate_impact(self):
        return self.base_rate(privileged=False) / self.base_rate(privileged=True)

    def statistical_parity_difference(self):
        return self.base_rate(privileged=False) - self.base_rate(privileged=True)

    def consistency(self, n_neighbors: int = 5):
        r"""Individual fairness metric from [1]_ that measures how similar the
        labels are for similar instances.
        .. math::
            1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i -
            \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|
        Args:
            n_neighbors (int, optional): Number of neighbors for the knn
                computation.
        References:
            .. [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
                "Learning Fair Representations,"
                International Conference on Machine Learning, 2013.
        """

        X = self.dataset.features
        num_samples = X.shape[0]
        y = self.dataset.labels

        nbrs = NearestNeighbors().fit(X)
        _, indices = nbrs.kneighbors(X)

        # compute consistency score
        consistency = 0.0
        for i in range(num_samples):
            consistency += np.abs(y[i] - np.mean(y[indices[i]]))
        consistency = 1.0 - consistency / num_samples

        return consistency[0]

    def smoothed_base_rate(self, concentrate: float = 1.0):
        num_classes = len(np.unique(self.dataset.labels))
        dirichlet_alpha = 1.0 / num_classes
        intersect_groups = list(
            self.df.groupby(self.protected_attribute_names).groups.keys()
        )
        num_intersects = len(intersect_groups)

        # make counts total
        result = []
        for intgrp in intersect_groups:
            tdf = self.df.copy()

            # calculate total count
            for idx, value in enumerate(intgrp):
                att_name = self.protected_attribute_names[idx]
                tdf = tdf[tdf[att_name] == value].copy()
            total = len(tdf)

            # calculate positive count
            for name in self.label_names:
                tdf = tdf[tdf[name] == self.favorable_label].copy()
            pos = len(tdf)

            result.append((pos + dirichlet_alpha) / (total + concentrate))
        return result

    def smoothed_empirical_differential_fairness(self, concentration: float = 1.0):
        sbr = self.smoothed_base_rate(concentrate=concentration)

        def pos_ratio(i, j):
            return abs(np.log(sbr[i]) - np.log(sbr[j]))

        def neg_ratio(i, j):
            return abs(np.log(1 - sbr[i]) - np.log(1 - sbr[j]))

        # overall DF of the mechanism
        return max(
            max(pos_ratio(i, j), neg_ratio(i, j))
            for i in range(len(sbr))
            for j in range(len(sbr))
            if i != j
        )


class ClassificationMetric(DataMetric):
    def __init__(self, dataset, prediction_vector, target_label_name):
        super(ClassificationMetric, self).__init__(dataset)
        self.prediction_vector = prediction_vector
        self.target_label = target_label_name
        self.df = self.df.iloc[0 : len(self.prediction_vector), :]
        self.conf_mat = self.confusion_matrix()
        self.performance = self.performance_measures()

    def confusion_matrix(self, privileged=None):
        df = self.df.copy()
        if not len(df) == len(self.prediction_vector):
            raise ValueError

        df["Prediction"] = self.prediction_vector
        df = self.select_data_by_privilege(selected_data=df, privileged=privileged)

        # True prediction
        tpred = df[df[self.target_label] == df["Prediction"]].copy()
        # True Positive
        tp = len(tpred[tpred[self.target_label] == self.favorable_label])
        # True Negative
        tn = len(tpred[tpred[self.target_label] == self.unfavorable_label])

        # False prediction
        fpred = df[df[self.target_label] != df["Prediction"]].copy()
        # False Positive
        fp = len(fpred[fpred[self.target_label] == self.unfavorable_label])
        # False Negative
        fn = len(fpred[fpred[self.target_label] == self.favorable_label])

        return dict(TP=tp, TN=tn, FP=fp, FN=fn)

    def performance_measures(self, privileged: bool = False):
        conf_mat = self.confusion_matrix(privileged=privileged)

        tp = conf_mat["TP"]
        tn = conf_mat["TN"]
        fp = conf_mat["FP"]
        fn = conf_mat["FN"]

        p = self.num_positive(privileged=privileged)
        n = self.num_negative(privileged=privileged)

        return dict(
            TPR=tp / p if p > 0.0 else 0.0,
            TNR=tn / n if n > 0.0 else 0.0,
            FPR=fp / n if n > 0.0 else 0.0,
            FNR=fn / p if p > 0.0 else 0.0,
            PPV=tp / (tp + fp) if (tp + fp) > 0.0 else 0.0,
            NPV=tn / (tn + fn) if (tn + fn) > 0.0 else 0.0,
            FDR=fp / (fp + tp) if (fp + tp) > 0.0 else 0.0,
            FOR=fn / (fn + tn) if (fn + tn) > 0.0 else 0.0,
            ACC=(tp + tn) / (p + n) if (p + n) > 0.0 else 0.0,
        )

    def error_rate(self):
        return 1 - self.performance["ACC"]

    def average_odds_difference(self, is_abs: bool = False):
        pri_perf = self.performance_measures(privileged=True)
        unpri_perf = self.performance_measures(privileged=False)

        diff_fpr = unpri_perf["FPR"] - pri_perf["FPR"]
        diff_tpr = unpri_perf["TPR"] - pri_perf["TPR"]

        if is_abs:
            return 0.5 * (np.abs(diff_fpr) + np.abs(diff_tpr))
        return 0.5 * (diff_fpr + diff_tpr)

    """
    def average_abs_odds_difference(self):
        pri_perf = self.performance_measures(privileged=True)
        unpri_perf = self.performance_measures(privileged=False)

        diff_FPR = UnpriPerfM["FPR"] - PriPerfM["FPR"]
        diff_TPR = UnpriPerfM["TPR"] - PriPerfM["TPR"]

        return 0.5 * (np.abs(diff_FPR) + np.abs(diff_TPR))
    """

    def selection_rate(self, privileged: bool = False):
        conf_mat = self.confusion_matrix(privileged=privileged)

        num_pred_positives = conf_mat["TP"] + conf_mat["FP"]
        num_instances = (
            conf_mat["TP"] + conf_mat["FP"] + conf_mat["TN"] + conf_mat["FN"]
        )

        if num_instances == 0:
            return 0
        else:
            return num_pred_positives / num_instances

    def disparate_impact(self):
        denom = self.selection_rate(privileged=False)
        nom = self.selection_rate(privileged=True)
        if nom == 0:
            return 0
        else:
            return denom / nom

    def statistical_parity_difference(self):
        denom = self.selection_rate(privileged=False)
        nom = self.selection_rate(privileged=True)
        return denom - nom

    def generalized_entropy_index(self, alpha=2):
        pred_df = self.df.copy()
        pred_df["Prediction"] = self.prediction_vector

        y_pred = (
            (pred_df["Prediction"] == self.favorable_label)
            .to_numpy()
            .astype(np.float64)
        )
        y_true = (
            (self.df[self.target_label] == self.favorable_label)
            .to_numpy()
            .astype(np.float64)
        )
        b = 1 + y_pred - y_true

        if alpha == 1:
            # moving the b inside the log allows for 0 values
            result = np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
        elif alpha == 0:
            result = -np.mean(np.log(b / np.mean(b)) / np.mean(b))
        else:
            result = np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))
        return result

    def theil_index(self):
        r"""The Theil index is the :meth:`generalized_entropy_index` with
        :math:`\alpha = 1`.
        """
        return self.generalized_entropy_index(alpha=1)

    def equal_opportunity_difference(self):
        r""":math:`TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}`"""

        ## TPR_unprivileged
        perfm_unpriv = self.performance_measures(privileged=False)
        tpr_unpriv = perfm_unpriv["TPR"]

        ## TPR_privileged
        perfm_priv = self.performance_measures(privileged=True)
        tpr_priv = perfm_priv["TPR"]
        return tpr_unpriv - tpr_priv


def compute_tsne(dataset: aifData, sample_size: int = 10):
    print("T-SNE analysis start")
    priv_val = dataset.privileged_protected_attributes[0][0]
    unpriv_val = dataset.unprivileged_protected_attributes[0][0]

    df = dataset.convert_to_dataframe()[0]
    df_priv = df.loc[df[dataset.protected_attribute_names[0]] == priv_val]
    df_unpriv = df.loc[df[dataset.protected_attribute_names[0]] == unpriv_val]
    ds_priv = aifData(
        df=df_priv,
        label_name=dataset.label_names[0],
        favorable_classes=[dataset.favorable_label],
        protected_attribute_names=dataset.protected_attribute_names,
        privileged_classes=dataset.privileged_protected_attributes,
    )
    ds_unpriv = aifData(
        df=df_unpriv,
        label_name=dataset.label_names[0],
        favorable_classes=[dataset.favorable_label],
        protected_attribute_names=dataset.protected_attribute_names,
        privileged_classes=dataset.privileged_protected_attributes,
    )

    priv_sample = random.sample(ds_priv.features.tolist(), k=sample_size)
    priv_sample = np.array(priv_sample)
    unpriv_sample = random.sample(ds_unpriv.features.tolist(), k=sample_size)
    unpriv_sample = np.array(unpriv_sample)

    # T-SNE analysis
    tsne_priv = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=5
    ).fit_transform(priv_sample)
    tsne_priv = tsne_priv.tolist()

    tsne_unpriv = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=5
    ).fit_transform(unpriv_sample)
    tsne_unpriv = tsne_unpriv.tolist()

    return tsne_priv, tsne_unpriv


def get_baseline_result(trainset: aifData, testset: aifData, baseline: str = "svm"):
    if baseline == "svm":
        baseline = svm.SVC(random_state=777)

    baseline.fit(trainset.features, trainset.labels.ravel())
    pred = baseline.predict(testset.features)
    return pred


def get_metrics(dataset: aifData):
    """
    dataset: load_preproc 등을 거친 이후의 데이터셋
    """
    tsne_priv, tsne_unpriv = compute_tsne(dataset)

    data_metric = DataMetric(dataset=dataset)
    traindata, testdata = dataset.split([0.7], shuffle=True)
    cls_metric = ClassificationMetric(
        dataset=dataset,
        prediction_vector=get_baseline_result(trainset=traindata, testset=testdata),
        target_label_name=dataset.label_names[0],
    )
    perfm = cls_metric.performance_measures()

    metrics = {
        "data": {
            "protected": dataset.protected_attribute_names[0],
            "privileged": {
                "num_negatives": data_metric.num_negative(privileged=True),
                "num_positives": data_metric.num_positive(privileged=True),
                "TSNE": tsne_priv,
            },
            "unprivileged": {
                "num_negatives": data_metric.num_negative(privileged=False),
                "num_positives": data_metric.num_positive(privileged=False),
                "TSNE": tsne_unpriv,
            },
            "base_rate": round(data_metric.base_rate(), 3),
            "statistical_parity_difference": round(
                data_metric.statistical_parity_difference(), 3
            ),
            "consistency": round(data_metric.consistency(), 3),
        },
        "performance": {
            "recall": round(perfm["TPR"], 3),
            "true_negative_rate": round(perfm["TNR"], 3),
            "false_positive_rate": round(perfm["FPR"], 3),
            "false_negative_rate": round(perfm["FNR"], 3),
            "precision": round(perfm["PPV"], 3),
            "negative_predictive_value": round(perfm["NPV"], 3),
            "false_discovery_rate": round(perfm["FDR"], 3),
            "false_omission_rate": round(perfm["FOR"], 3),
            "accuracy": round(perfm["ACC"], 3),
        },
        "classify": {
            "error_rate": round(cls_metric.error_rate(), 3),
            "average_odds_difference": round(cls_metric.average_odds_difference(), 3),
            "average_abs_odds_difference": round(
                cls_metric.average_odds_difference(is_abs=True), 3
            ),
            "selection_rate": round(cls_metric.selection_rate(), 3),
            "disparate_impact": round(cls_metric.disparate_impact(), 3),
            "statistical_parity_difference": round(
                cls_metric.statistical_parity_difference(), 3
            ),
            "generalized_entropy_index": round(
                cls_metric.generalized_entropy_index(), 3
            ),
            "theil_index": round(cls_metric.theil_index(), 3),
            "equal_opportunity_difference": round(
                cls_metric.equal_opportunity_difference(), 3
            ),
        },
    }
    return metrics
