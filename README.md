# MSIT AI Fair(MAF)

MSIT AI Fair(MAF)는 과학기술정보통신부 “점차 강화되고 있는 윤리 정책에 발맞춰 유연하게 진화하는 인공지능 기술 개발 연구(2022~2026)“ 국가과제의 일환으로, 인공지능(AI)의 공정성을 진단하고 편향성을 교정하는 진단 시스템입니다. 과거 “인공지능 모델과 학습데이터의 편향성 분석-탐지-완화-제거 지원 프레임워크 개발(2019-2022)” 국가과제 결과물의 연장선으로, 지속적으로 확장·개발되고 있습니다.

MAF는 데이터 편향성과 알고리즘 편향성을 측정 및 완화하는 것을 목표로 합니다. MAF는 IBM에서 공개한 AI Fairness 360(AIF360)의 브랜치로 시작하여 AIF360의 기본 기능을 담고 있으며, 과제 수행 기간 중 컨소시엄 내에서 개발된 편향성 완화 알고리즘의 추가, 지원 데이터 형식 추가, CPU 환경 지원 추가 등의 기능을 계속 확장하고 있습니다.

MAF 패키지는 python 환경에서 사용할 수 있습니다.

MAF 패키지에는 다음이 포함됩니다.
1. 모델에 대한 메트릭 세트 및 메트릭에 대한 설명
2. 데이터 세트 및 모델의 편향을 완화하는 알고리즘
      * 연구소 알고리즘은 음성, 언어, 금융, 의뢰 시스템, 의료, 채용, 치안, 광고, 법률, 문화, 방송 등 광범위한 분야에서 활용하기 위해 설계되었습니다.

확장 가능성을 두고 패키지를 개발하였으며 지속적으로 개발 및 업데이트를 진행 중입니다.

# Framework Outline
MAF는 크게 algorithms, benchmark, metric의 세 파트로 이루어져 있습니다.
algorithms 파트는 편향성 완화와 관련된 다양한 알고리즘이 포함되어있으며, AIF360의 분류를 따라 알고리즘을 Pre/In/Post Processing 3가지로 분류하고 있습니다. benchmark 파트는 편향성 완화와 관련된 각 benckmark를 테스트 해볼 수 있는 모듈들로 구성되어있으며, metric 파트는 편향성 측정 지표와 관련한 모듈들로 구성되어있습니다.

algorithms, benchmark, metric 각 파트의 구성은 다음과 같습니다.

## Algorithms
AIF360의 알고리즘 및 편향성 완화와 관련한 최신 연구들을 포함하고 있습니다.
### Pre-Processing Algorithms
* AIF360
  * Disparate Impact Remover
  * Learning Fair Representation
  * Reweighing
  * Optim Preproc
  * Learning Fair Representation
* SOTA Algorithm
  * Co-occurrence-bias [📃paper](https://aclanthology.org/2023.findings-emnlp.518.pdf) [💻 code](https://github.com/CheongWoong/impact_of_cooccurrence)
  * Fair Streaming PCA [📃paper](https://arxiv.org/abs/2310.18593) [💻 code](https://github.com/HanseulJo/fair-streaming-pca/?tab=readme-ov-file)
  * Representative Heuristic
  * Fair Batch [📃paper](https://arxiv.org/abs/2012.01696) [💻 code](https://github.com/yuji-roh/fairbatch) (to be updated)

### In-Processing Algorithms
* AIF360
  * Gerry Fair Classifier (to be updated)
  * Meta Fair Classifier
  * Prejudice Remover
  * Exponentiated Gradient Reduction
* SOTA Algorithm
  * ConCSE [📃paper](https://arxiv.org/abs/2409.00120) [💻 code](https://github.com/jjy961228/ConCSE?tab=readme-ov-file)
  * INTapt [📃paper](https://arxiv.org/abs/2305.16371)
  * Fair Dimension Filtering
  * Fairness Through Matching [💻 code]("https://github.com/kwkimonline/FTM)
  * Fair Feature Distillation [📃paper](https://arxiv.org/abs/2106.04411) [💻 code](https://github.com/DQle38/Fair-Feature-Distillation-for-Visual-Recognition)
  * SLIDE [📃paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608022002891) [💻 code](https://github.com/kwkimonline/SLIDE?tab=readme-ov-file)
  * sIPM-LFR  [📃paper](https://arxiv.org/abs/2202.02943) [💻 code](https://github.com/kwkimonline/sIPM-LFR) (to be updated)
  * Learning From Fairness [📃paper](https://arxiv.org/abs/2007.02561) [💻 code](https://github.com/alinlab/LfF)
  * Fairness VAE [📃paper](https://arxiv.org/abs/2007.03775) [💻 code](https://github.com/sungho-CoolG/Fairness-VAE) (to be updated)
  * Kernel Density Estimator [📃paper](https://proceedings.neurips.cc/paper/2020/hash/ac3870fcad1cfc367825cda0101eee62-Abstract.html) [💻 code](https://github.com/Gyeongjo/FairClassifier_using_KDE) (to be updated)

### Post-Processing Algorithms
* AIF360
  * Calibrated EqOdds
  * Equalized Odds
  * Reject Option Classifier

## Benchmark
편향성과 관련한 benchmark에 대한 연구들을 포함하고 있습니다.

* KoBBQ [📃paper](https://arxiv.org/abs/2307.16778) [💻 code](https://github.com/naver-ai/KoBBQ?tab=readme-ov-file)
* CREHate [📃paper](https://arxiv.org/abs/2308.16705) [💻 code](https://github.com/nlee0212/CREHate)

## Metric
편향성을 측정할 수 있는 metric과 관련한 연구들을 포함하고 있습니다.

* Latte [📃paper](https://arxiv.org/pdf/2402.06900v3)

MAF에서는 AIF360에 제시된 편향성 관련 metric도 함께 지원하고 있습니다.
### Data metrics
* Number of negatives (privileged)
* Number of positives (privileged)
* Number of negatives (unprivileged)
* Number of positives (unprivileged)
* Base rate
* Statistical parity difference
* Consistency

### Classification metrics
* Error rate
* Average odds difference
* Average abs odds difference
* Selection rate
* Disparate impact
* Statistical parity difference
* Generalized entropy index
* Theil index
* Equal opportunity difference

# Setup
Supported Python Configurations:

| OS      | Python version |
| ------- | -------------- |
| macOS   | 3.8 – 3.11     |
| Ubuntu  | 3.8 – 3.11     |
| Windows | 3.8 – 3.11     |

### (Optional) Create a virtual environment
MAF의 원활한 구동을 위해서는 특정 버전의 패키지들이 필요합니다. 시스템의 다른 프로젝트와 충돌할 수 있으므로 anaconda 가상 환경 혹은 docker를 권장드립니다.

### Installation
1. 이 저장소의 최신 버전을 복제합니다.
```bash
git clone https://github.com/konanaif/MAF2024.git
```

2. 필요한 패키지들을 설치합니다.
```bash
conda install --file requirements.txt
```
