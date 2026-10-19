

### Implementation and validation

The evaluation package implements LiMP and seven baseline classifiers: RF, XGB, SVM, BERT, DistilBERT, T5, and DeBERTa. It includes data preprocessing, training, checkpoint selection, prediction export, performance evaluation, resource monitoring, statistical comparisons, ablation experiments, and explanation modules.

Validation was conducted in a CPU-based Linux environment. Installed package versions are recorded in `tested_versions.json`. Python compilation, command-line argument parsing, and ten automated tests passed. These tests covered metric calculations, Brier-score scaling, linguistic features, training-only normalization, split isolation, Holm adjustment, paired statistical calculations, LiMP component configurations, attention masking, prototype routing, gradient propagation, frozen-backbone behaviour, and checkpoint persistence.

The supplied dataset was loaded and checked successfully. It contained 12,883 claims across three credibility classes, with no exact duplicate claims after text normalization. Stratified fitting, validation, and test partitions were generated and recorded.

RF, SVM, and XGB were trained and evaluated on the supplied dataset using a functional verification configuration. This configuration used a maximum of 5,000 TF–IDF features, 20 trees for the tree-based classifiers, one training seed, and one CPU job. Held-out predictions were exported successfully. Comparative reports included confusion matrices, ROC curves, calibration curves, confidence intervals, Brier-score and log-loss analyses, resource measurements, and an Excel results workbook.

Transformer classification wrappers were validated using compact offline configurations of BERT, DistilBERT, T5, and DeBERTa. LiMP training and diagnostic export were verified through a compact offline integration run. The resulting diagnostics covered attention weights, linguistic gates, residual fusion, prototype similarities, routing behaviour, representation projections, boundary-refinement transitions, class-wise precision–recall curves, and error analysis.

SHAP and LIME explanation modules were executed on a fitted SVM using three selected claims. Token contributions, explanation figures, attribution-removal curves, and faithfulness measurements were generated. Six-topic NMF analysis and top-five-keyword visualizations were also completed using the supplied dataset.

The resource monitor records process memory and supports PyTorch GPU-memory measurements. Process monitoring was adapted to the validation environment’s PID namespace. Single-job execution resolved the environment’s multiprocessing limitation.

The completed checks establish implementation functionality and successful execution of the tested evaluation paths. Full pretrained-transformer training, the 77-epoch multi-seed experiments, complete retrained ablation suites, LiMP checkpoint explanations, and T1200 GPU benchmarking remain outstanding. Friedman and Nemenyi comparisons require additional independent datasets. Consequently, the validation described here establishes software functionality; final LiMP performance and comparative significance require the corresponding completed experiments.