# Optimization analysis for string similarity choice

This is an auxiliary pipeline to reproduce the results we showed in the report for th eoptimization of the string similarity choice. We tested two factors: the string similarity options (levenshtein, damerau, jaccard, cosine, jaro_winkler, longest_common_subsequence, metric_lcs, ngram, optimal_string_alignment, overlap_coefficient, qgram, sorensen_dice) and the aplication of string normalization (to lower case). We tried two distiinct approaches: maximize the distance value, and exeecute for similarity minimization.

## Execution
The package dependencies are covered by the conda environment configuration file in the root of this repository (environment.yml).

Running analysis: `python3 optimization_choice_string_metric.py _outFolder_` , where \_outFolder\_ is the path to a custom output folder. If this parameter is empty, it will save in the default path "./out_ss_choice_optimization"

## Outputs explanation
In the example "./out_ss_choice_optimization" folder, you will find the following output items:
- goldds_labelled_mapping_nct_pubmed.tsv : table mapping the clinical trial identifier t the pubmed id, associating both to the text annotated and the respective entity.

- fast_gold_results_test_validation.tsv : Initial pairwise similarity calculation to obtain the combination pairs between the clinical trials processed data items per entity and the candidate annotations for these entities.

- optimization/by_distance_best_params.pkl - pickle file containing the best combination of the variables in the optimization test, using distance minimization.

- optimization/by_similarity_best_params.pkl - pickle file containing the best combination of the variables in the optimization test, using similarity maximization.

- optimization/fast_gold_cosine_results_validation.tsv - table with the pairwise similarity calculation between annotation candidates and clinical trial items for the same entities using the best string similarity found by the optimization. It also adds the score values with and without normalization.

- optimization/grouped_fast_gold_cosine_results_validation.tsv - the table above grouping by the maximum similarity score, and eliminating the combinations that did not yield the maximum score for each annotation.

- optimization/cosine_gold_grouped_distribution_scoresim.png - plot showing the distribution of similarity values (boxplot + data points) with normalization using the best metric found by the optimization

- cosine_gold_all_distribution_scoresim.png - plot showing the distribution of similarity values (only boxplot) with their comparison with and without string normalization, also associating a p-value.

