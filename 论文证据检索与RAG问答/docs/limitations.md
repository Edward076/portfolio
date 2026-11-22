# Limitations and Outstanding Work

- The primary evaluation covers all 281 Qasper `validation` papers, but not the official test split or an independent benchmark.
- The original reported runs used Python 3.13.16. Separate Python 3.11.9 reruns cover all four methods for full validation and the 50-paper pilot, K=20 candidates, bootstrap analysis, error classification, and development-set chunking/Top-K ablations. Full-validation metrics, pilot rankings, and development ablation rows reproduce the saved references; remote CI is configured but has not run on GitHub.
- Exact normalized evidence matching does not map every annotation (86.4% of full-validation evidence items map). The 382 unmatched annotations remain recorded and are omitted from retrieval scoring.
- Seventy-five full-validation questions contain both answerable and unanswerable annotator variants. They are retained; generation abstention scoring needs a declared aggregation rule.
- The full evaluation has 276 scored paper clusters. The smaller 50-paper pilot (163 scored questions) is exploratory and reported separately.
- Subgroup metrics, especially yes/no (102 scored questions) and mixed-answerability (65), are less precise than the overall comparison and should be treated as descriptive.
- The development subset ablations contain 36 scored questions; their estimates are exploratory.
- Exact text matching, even after normalization, can miss evidence that is paraphrased, merged, split differently, or represented as figures/tables.
- CPU timing is specific to this host and this software/cache state; it is not a deployment throughput benchmark.
- The separate local pilot measures answer and citation alignment and token use for one small quantized model on 20 selected questions. Its citation constraint enforces context membership. Semantic support and paid API costs remain unmeasured.
- No public GitHub repository has been created or pushed. Local work is kept in an independent repository directory.
