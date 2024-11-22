import MAF2024.algorithms.preprocessing as preprocessing
import MAF2024.algorithms.inprocessing as inprocessing
from MAF2024.algorithms.inprocessing.concse import mitigate_concse
from MAF2024.algorithms.inprocessing.INTapt.intapt import mitigate_intapt
import MAF2024.algorithms.postprocessing as postprocessing

from MAF2024.benchmark.crehate.crehate_demo import check_hatespeech
from MAF2024.benchmark.kobbq.kobbq_demo import check_korean_bias, KoBBQArguments

from MAF2024.metric.latte.check_toxicity import check_toxicity
from MAF2024.metric.metric import get_metrics

from MAF2024.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_compas,
    load_preproc_data_german,
    load_preproc_data_adult,
    load_preproc_data_celeba,
    load_preproc_data_pubfig,
)

from collections import OrderedDict
import os
import numpy as np

os.environ["PYTHONPATH"] = "/workspace"


def check_type_for_metric(metrics: dict):
    assert type(metrics["data"]) == type({})
    assert type(metrics["performance"]) == type({})
    assert type(metrics["classify"]) == type({})


def test_metric():
    print("check metric for compas")
    data = load_preproc_data_compas()
    check_type_for_metric(get_metrics(data))

    print("check metric for german")
    data = load_preproc_data_german()
    check_type_for_metric(get_metrics(data))

    print("check metric for adult")
    data = load_preproc_data_adult()
    check_type_for_metric(get_metrics(data))

    print("check metric for pubfig")
    data = load_preproc_data_pubfig()
    check_type_for_metric(get_metrics(data))

    print("check metric for celeba")
    data = load_preproc_data_celeba()
    check_type_for_metric(get_metrics(data))


def check_type_for_tabular_algorithm(algorithm_obj):
    original_metric, mitigated_metric = algorithm_obj.run()
    for k in original_metric:
        if k == "protected":
            assert type(original_metric[k]) == str
        else:
            assert type(original_metric[k]) == np.float64
            assert type(mitigated_metric[k]) == np.float64


def check_type_for_image_algorithm(algorithm_obj):
    original_metric, mitigated_metric = algorithm_obj.run()
    for k in original_metric:
        if k == "protected":
            assert type(original_metric[k]) == str
        else:
            assert (type(original_metric[k]) == np.float64) or (
                type(original_metric[k]) == float
            )
            assert (type(mitigated_metric[k]) == np.float64) or (
                type(mitigated_metric[k]) == float
            )


def test_image_algorithm(data_name: str = "pubfig"):
    fdf = inprocessing.fair_dimension_filtering.FairDimFilter(dataset_name="celeba")
    check_type_for_image_algorithm(fdf)

    ffd = inprocessing.fair_feature_distillation.FairFeatureDistillation(
        dataset_name=data_name
    )
    check_type_for_image_algorithm(ffd)


def test_audio_algorithm():
    result = mitigate_intapt()
    for k in result:
        assert type(result[k]) == float


def test_tabular_algorithm(data_name: str = "adult"):
    dirm = preprocessing.disparate_impact_remover.DisparateImpactRemover(
        dataset_name=data_name, protected="sex", repair_level=1.0
    )
    check_type_for_tabular_algorithm(dirm)

    lfr = preprocessing.learning_fair_representation.LearningFairRepresentation(
        dataset_name=data_name, protected="race"
    )
    check_type_for_tabular_algorithm(lfr)

    optimpreproc = preprocessing.optim_preproc.OptimPreproc(
        dataset_name=data_name, protected="sex"
    )
    check_type_for_tabular_algorithm(optimpreproc)

    rw = preprocessing.reweighing.Reweighing(dataset_name=data_name, protected="sex")
    check_type_for_tabular_algorithm(rw)

    fairpca = preprocessing.fairpca.MeanCovarianceMatchingFairPCAWithClassifier(
        dataset_name=data_name, protected="sex"
    )
    check_type_for_tabular_algorithm(fairpca)

    slide = inprocessing.slide.SlideFairClassifier(
        dataset_name=data_name, protected="sex"
    )
    check_type_for_tabular_algorithm(slide)

    ftm = inprocessing.ftm.FTMFairClassifier(dataset_name=data_name, protected="sex")
    check_type_for_tabular_algorithm(ftm)

    egr = inprocessing.exponentiated_gradient_reduction.ExponentiatedGradientReduction(
        dataset_name=data_name, protected="sex"
    )
    check_type_for_tabular_algorithm(egr)

    prejremover = inprocessing.prejudice_remover.PrejudiceRemover(
        dataset_name=data_name, protected="sex"
    )
    check_type_for_tabular_algorithm(prejremover)

    mfc = inprocessing.meta_classifier.MetaFairClassifier(
        dataset_name=data_name, protected="sex"
    )
    check_type_for_tabular_algorithm(mfc)

    ceo = postprocessing.calibrated_eq_odds.CalibratedEqOdds(
        dataset_name=data_name, protected="sex"
    )
    check_type_for_tabular_algorithm(ceo)

    eo = postprocessing.equalize_odds.EqOdds(dataset_name=data_name, protected="sex")
    check_type_for_tabular_algorithm(eo)

    roc = postprocessing.reject_option_classification.RejectOptionClassifier(
        dataset_name="compas", protected="sex"
    )
    check_type_for_tabular_algorithm(roc)


if __name__ == "__main__":
    test_metric()
    test_tabular_algorithm()
    test_image_algorithm()
    test_audio_algorithm()
