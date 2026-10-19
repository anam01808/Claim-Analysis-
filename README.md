# LiMP: Credibility Classification and Evaluation

The package implements LiMP and seven baseline classifiers: RF, XGB, linear SVM, BERT, DistilBERT, T5-base, and DeBERTa-v3-base. It includes the dataset, source code, training procedures, evaluation modules, explanation methods, statistical comparisons, and resource monitoring.

## 1. Installation

Python 3.10–3.12 is recommended. Extract the archive and open a terminal in the package directory.

Create a virtual environment:

```text
python -m venv .venv
```

Activate the environment on Windows:

```text
.venv\Scripts\activate
```

Activate the environment on Linux or macOS:

```text
source .venv/bin/activate
```

Install PyTorch ≥2.6 and <3 using the appropriate command for the operating system, NVIDIA driver, and CUDA configuration:

https://pytorch.org/get-started/locally/

For CPU execution:

```text
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies and run the automated checks:

```text
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

Pretrained transformer weights are downloaded from Hugging Face during initial execution. Models run sequentially. Completed runs are retained and skipped unless `--overwrite` is specified. Interrupted model runs restart training, and failed runs record diagnostic information in `FAILED.txt`.

## 2. Model Training

Run RF to verify the installation and dataset configuration:

```text
python run.py train --data data/Claim_Credibility.xlsx --out results_frozen --models RF --seeds 42
python run.py report --out results_frozen --seed 42
```

Train all eight models with a maximum of 77 epochs:

```text
python run.py train --data data/Claim_Credibility.xlsx --out results_frozen --models RF XGB SVM BERT DistilBERT T5 DeBERTa LiMP --seeds 42 --epochs 77 --mode frozen --batch-size 1 --accumulation 16
python run.py report --out results_frozen --seed 42
```

The `frozen` protocol retains fixed transformer-backbone parameters and trains the classification heads and LiMP modules. This protocol reduces optimizer-memory requirements and applies consistently across transformer models. Classical classifiers execute on the CPU.

Full encoder fine-tuning uses the `full` protocol and a separate output directory:

```text
python run.py train --data data/Claim_Credibility.xlsx --out results_full --models BERT DistilBERT T5 DeBERTa LiMP --mode full --batch-size 1 --accumulation 16 --epochs 77
```

Early stopping uses validation macro-F1 with a default patience of 10 epochs. Add `--patience 0` to execute all 77 epochs. The checkpoint with the highest validation macro-F1 is retained.

Temperature calibration is fitted using validation labels:

```text
python run.py train --data data/Claim_Credibility.xlsx --out results_calibrated --models LiMP --temperature
```

## 3. Complete Study Execution

The study launcher supports eight models, five optimization seeds, four experimental suites, explanation analysis, topic modelling, and report generation:

```text
python run_study.py --data data/Claim_Credibility.xlsx --out results_frozen --mode frozen --epochs 77 --seeds 42 13 21 87 100 --suites --explanations --topics
```

The first listed seed is used for the primary report. Individual reporting and explanation commands are:

```text
python run.py report --out results_frozen --seed 42 --bootstrap 2000
python run.py explain --out results_frozen --seed 42 --samples 6
```

Paired holdout comparisons use a prespecified primary seed. Across-seed standard deviations summarize optimization variability on the fixed data partition.

## 4. Dataset and Partitioning

The dataset is stored in `data/Claim_Credibility.xlsx` and contains 12,883 claims:

| Class | Number of claims |
|---|---:|
| Credible | 5,772 |
| Non-Credible | 5,779 |
| Questionable | 1,332 |
| Total | 12,883 |

The required columns are `Claim` and `Credibility Label`. Claim text supplies the prediction input. ID, Topic, Source Type, and Evidence Status are excluded from model predictors. Topic annotations are retained for descriptive analysis.

Stratified partitioning uses split seed 42:

- Held-out test set: 20%, comprising 2,577 claims.
- Fitting and validation sets: the remaining 80%, partitioned using `--val-fraction .2`.
- Overall allocation: approximately 64% fitting, 16% validation, and 20% testing.
- Identical partitions across models, optimization seeds, and experimental variants.
- TF–IDF vocabulary and linguistic-feature normalization fitted on fitting claims.
- Empty-text removal and normalized duplicate checking before partitioning.
- Dataset counts, file checksum, and preprocessing records stored in `audit.json`.

Validation identified no exact duplicate claims after normalization.

The class mapping is:

```text
0 = Credible
1 = Non-Credible
2 = Questionable
```

Three-class evaluation uses 3 × 3 confusion matrices. Sample-level TP, FP, FN, and TN outcomes use Credible as the positive class and combine the remaining labels as the negative class.

## 5. LiMP Architecture

The model specification is implemented in `limp/model.py` and `limp/data.py`.

1. DeBERTa-v3-base produces contextual token representations, projected to 256 dimensions.
2. Six linguistic feature groups contain four deterministic numerical features each, with 64-dimensional group projections.
3. Six sigmoid gates regulate the group representations. Their 384-dimensional concatenation is projected to 256 dimensions.
4. Semantic attention pooling uses a 128-dimensional scoring layer.
5. Four-head linguistic-conditioned attention queries token representations using the linguistic vector.
6. Semantic, conditioned, linguistic, and absolute-difference representations form a 1,024-dimensional fusion input, projected to 256 dimensions.
7. Absolute differences and element-wise products produce a 512-dimensional disagreement input, projected to 256 dimensions.
8. A 256-dimensional sigmoid gate controls residual disagreement fusion.
9. Three prototypes per class provide nine prototype representations. Cosine similarities support routing with temperature 0.1 and contribute class-level prototype scores.
10. Prototype learning incorporates a pull-loss coefficient of 0.01 and a within-class diversity coefficient of 0.001.
11. Boundary refinement combines the representation and three class probabilities through a 259 → 128 → 256 network.
12. Training uses weighted cross-entropy, AdamW, encoder learning rate 2 × 10⁻⁵ in full mode, added-module learning rate 1 × 10⁻⁴, weight decay 0.01, gradient clipping at 1, and 10% linear warm-up followed by decay.

Additional settings include maximum sequence length 128, default module dropout 0.2, and automatic mixed precision on compatible CUDA devices.

Linguistic features use explicit numerical definitions and word lists. Feature statistics are recorded in `feature_definitions.csv`. Group ablations remove the selected group before subsequent gating and fusion.

T5 uses an encoder–decoder sequence-classification model. SVM uses a linear decision function with calibrated probabilities. RF, XGB, and SVM use unigram and bigram TF–IDF representations. Training arguments and model configurations are retained with each run.

## 6. Ablation and Sensitivity Analysis

Each experimental suite retrains LiMP variants using the established partitions and specified seeds:

```text
python run.py suite --data data/Claim_Credibility.xlsx --out results_frozen --kind component --seeds 42
python run.py suite --data data/Claim_Credibility.xlsx --out results_frozen --kind linguistic --seeds 42
python run.py suite --data data/Claim_Credibility.xlsx --out results_frozen --kind prototype --seeds 42
python run.py suite --data data/Claim_Credibility.xlsx --out results_frozen --kind hyperparameter --seeds 42
python run.py report --out results_frozen --seed 42
```

The suites cover:

- Component ablation: linguistic representation, group gating, conditioned attention, residual fusion, prototypes, and boundary refinement.
- Linguistic-feature ablation: the complete feature set and six leave-one-group-out configurations.
- Prototype sensitivity: one to five prototypes per class.
- Hyperparameter sensitivity: head learning rate, dropout, sequence length, and encoder learning rate in full mode.

Hyperparameter settings include head learning rates of 5 × 10⁻⁵ and 2 × 10⁻⁴, dropout values of 0.1 and 0.3, and sequence lengths of 64 and 256. Full-mode encoder learning rates include 1 × 10⁻⁵ and 3 × 10⁻⁵.

Sensitivity tables use validation scores for configuration selection. The selected configuration is fixed before confirmatory test evaluation.

## 7. Explanations and Attribution Faithfulness

Generate LiMP explanations using:

```text
python run.py explain --out results_frozen --model LiMP --seed 42 --samples 6 --shap-evals 512 --lime-samples 1000
```

The explanation module loads a trained checkpoint. SHAP estimates text-segment contributions using permutation sampling and a `[MASK]` replacement baseline. LIME fits a local positional-word surrogate. Linguistic features are recomputed after each text perturbation.

Explanations target the original predicted class. Examples are selected deterministically across the true classes.

Outputs include:

- Signed token-attribution heatmaps.
- Local contribution plots.
- LIME HTML explanations.
- Token-level attribution CSV files.
- SHAP additivity checks.
- Ranked and random deletion curves.
- Attribution-removal AUC.
- Comprehensiveness and sufficiency at 20% removal.

SHAP uses at least `2 * feature_count + 1` evaluations. LIME coefficients describe the fitted local surrogate. Attention weights are exported as separate model diagnostics. Shaded regions in deletion plots represent sample standard deviation.

## 8. Evaluation Plots and Tables

The workbook `results_frozen/report/All_Results_Tables.xlsx` consolidates report-level tables. Supporting predictions and measurements are retained in CSV and NPZ files.

Comparative plots include:

- Accuracy and macro-F1 with bootstrap 95% confidence intervals.
- Paired macro-F1 difference plots.
- Across-seed mean and standard deviation.
- Model confusion matrices.
- Combined micro-average one-versus-rest ROC curves.
- Top-label calibration curves.
- Brier-score and log-loss empirical cumulative distributions.
- Cumulative loss means in test-row order.
- Validation performance against accumulated training time.
- Recorded RAM and GPU-memory traces.

Classical classifiers are represented by their completed-fit endpoints in performance-versus-time plots.

LiMP-specific plots include:

1. Feature-group gating heatmap.
2. Semantic and linguistic-conditioned attention comparison.
3. Linguistic–semantic disagreement distribution.
4. Residual fusion gate heatmap.
5. Claim–prototype similarity heatmap.
6. Prototype routing and usage analysis.
7. Before-and-after signed class-margin analysis.
8. UMAP representation visualization.
9. SHAP and LIME token-attribution plots.
10. Attribution-removal curves.
11. Training and validation curves.
12. Class-wise precision–recall curves.

Additional outputs cover claim-length errors, sample predictions, classification outcomes, topic distributions, and the five highest-weight keywords per NMF topic.

The report supports the following tables:

1. Component ablation.
2. Linguistic-feature ablation.
3. Prototype sensitivity.
4. Hyperparameter sensitivity.
5. Boundary-refinement transitions.
6. Class-wise performance.
7. Paired statistical comparisons.
8. Attribution faithfulness.
9. Error analysis.
10. Computational efficiency.
11. Run stability.

Evaluation metrics include accuracy, macro precision, macro recall, macro-F1, multiclass MCC, top-label ECE, log loss, multiclass Brier score, and macro one-versus-rest ROC AUC.

The Brier score sums squared errors across the three classes and ranges from 0 to 2. Scores are stored as numerical fractions. ECE uses 15 bins; calibration figures use 10 bins.

## 9. Statistical Analysis

Paired comparisons align held-out predictions using identical row identifiers and reference labels.

The statistical table includes:

- Exact two-sided McNemar p-values.
- Holm-adjusted McNemar p-values across baseline comparisons.
- Class-stratified paired-bootstrap 95% confidence intervals for accuracy and macro-F1 differences.
- Paired accuracy differences.
- Corrected discordant odds ratios.
- Standardized paired correctness differences.
- Significance indicators at α = 0.05.

McNemar analysis evaluates paired correctness. Bootstrap analysis estimates uncertainty in the selected metric differences. Holm adjustment applies to the McNemar comparison family; bootstrap intervals are reported separately. These claim-level intervals describe uncertainty conditional on the fitted models and held-out sample.

Friedman, Kendall’s W, and Nemenyi critical-difference analysis are provided for comparisons across independent datasets. The input CSV contains:

```text
independent_dataset,model,score
```

Execute the analysis using:

```text
python run.py independent-datasets --csv independent_dataset_scores.csv --out independent_stats --confirm-independent
```

The module produces the Friedman statistic, associated p-value, Kendall’s W, average ranks, Nemenyi comparisons, and a critical-difference rank plot. The implementation requires at least three complete datasets and three models. Optimization seeds and credibility classes remain separate from the independent-dataset comparison.

## 10. Computational Resource Measurements

Resource monitoring records:

- RAM: combined resident-set size of the Python process and child processes, sampled every 0.5 seconds.
- GPU memory: current and peak PyTorch allocated and reserved memory.
- Time: preprocessing, model setup, accumulated training, validation, complete execution, and test inference.
- Storage: saved model-file size.
- Model size: total and trainable parameter counts.
- Throughput: evaluated samples per second.

Memory values use GiB, calculated as bytes divided by 2³⁰. Process-tree RSS can count shared pages more than once, and periodic sampling can miss brief peaks. PyTorch GPU-memory figures describe allocator usage rather than total board-wide memory consumption.

CUDA synchronization is applied around neural timing measurements. Neural inference includes data loading, device transfer, prediction, and diagnostic collection. Classical inference timing covers prediction after TF–IDF transformation. These measurement boundaries are retained when interpreting computational comparisons.

## 11. Topic Modelling and Prediction

Generate six-topic NMF analysis:

```text
python run.py topics --out results_frozen --topics 6
```

Predict credibility probabilities for a claim:

```text
python run.py predict --out results_frozen --model LiMP --seed 42 --text "president say new policy may affect public health"
```

TF–IDF and NMF components are fitted on fitting claims. The fitted topic model then generates descriptive assignments for the dataset. Claims with zero topic weight remain unassigned. Original topic tags are exported as multi-label claim counts.

## 12. Troubleshooting and Reproducibility

- GPU-memory management: use the frozen protocol, microbatch size 1, and a prespecified sequence length. Full fine-tuning requires sufficient GPU memory or CPU execution.
- Dependency errors: reinstall `requirements.txt` and verify the PyTorch installation.
- Windows runtime errors: verify compatible Python, PyTorch, and Microsoft Visual C++ runtime installations.
- Development checks: use `--debug-samples 30 --out debug_run --epochs 1` for neural execution checks.
- Output separation: retain distinct directories for frozen, full, calibrated, and development configurations.
- Configuration checks: completed-run skipping verifies saved training settings.
- Reproducibility records: inspect `environment.json`, `config.json`, `audit.json`, `splits.csv`, training histories, resource logs, predictions, checkpoints, and diagnostic exports.

Fixed seeds control the configured random-number generators. Hardware and numerical-library differences can affect exact numerical reproducibility.

## 13. Implementation Validation

`VALIDATION.md` records completed automated tests, dataset checks, classical-model verification runs, explanation-module execution, and offline transformer integration checks.

Full pretrained-transformer experiments, multi-seed study results, and target-hardware benchmarks are established through execution of the corresponding training and evaluation commands.

## 14. API References

- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [Hugging Face T5](https://huggingface.co/docs/transformers/model_doc/t5)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [SHAP text masker](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Text.html)
- [LIME text explanation](https://lime-ml.readthedocs.io/en/latest/lime.html)