# Run the whole pipeline from the project folder, with the virtual environment activated.
# Steps 03 must run before 04 (it writes results/oof_repeat0.csv).
$env:PYTHONUTF8 = "1"
python .\01_exploration.py       *> results_01.txt
python .\02_cox_ph.py            *> results_02.txt
python .\03_cv_evaluation.py     *> results_03.txt
python .\04_cohort_shift.py      *> results_04.txt

# Extension experiment: ALWAYS run the quick smoke test first, then the full run.
# python .\05_expression_experiment.py --smoke
# python .\05_expression_experiment.py *> results_05.txt
