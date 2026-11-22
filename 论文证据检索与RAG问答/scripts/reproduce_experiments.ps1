param(
    [string]$PythonExecutable = "python",
    [switch]$SkipDatasetPreparation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot

function Invoke-PythonStage {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    Write-Host "`n=== $Name ==="
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Name' failed with exit code $LASTEXITCODE."
    }
}

try {
    if (-not $SkipDatasetPreparation) {
        Invoke-PythonStage "Prepare deterministic Qasper samples" @(
            "scripts/prepare_dataset.py", "--seed", "42", "--dev-papers", "15", "--eval-papers", "50"
        )
    }

    $methods = @("bm25", "dense", "hybrid", "hybrid_reranked")
    foreach ($method in $methods) {
        Invoke-PythonStage "Held-out $method at K=5" @(
            "scripts/run_retrieval.py", "--split", "evaluation", "--method", $method,
            "--top-k", "5", "--chunking", "token-aware", "--max-tokens", "450",
            "--overlap-tokens", "50"
        )
    }

    foreach ($method in @("bm25", "dense")) {
        Invoke-PythonStage "Development $method original-paragraph chunking" @(
            "scripts/run_retrieval.py", "--split", "dev", "--method", $method, "--top-k", "5"
        )
        Invoke-PythonStage "Development $method token-aware chunking" @(
            "scripts/run_retrieval.py", "--split", "dev", "--method", $method,
            "--top-k", "5", "--chunking", "token-aware", "--max-tokens", "450",
            "--overlap-tokens", "50"
        )
    }

    foreach ($method in $methods) {
        Invoke-PythonStage "Development $method top-K ablation candidates" @(
            "scripts/run_retrieval.py", "--split", "dev", "--method", $method,
            "--top-k", "10", "--chunking", "token-aware", "--max-tokens", "450",
            "--overlap-tokens", "50", "--output-dir", "results/ablations/topk/$method"
        )
    }

    foreach ($method in $methods) {
        Invoke-PythonStage "Held-out $method top-20 error-analysis candidates" @(
            "scripts/run_retrieval.py", "--split", "evaluation", "--method", $method,
            "--top-k", "20", "--chunking", "token-aware", "--max-tokens", "450",
            "--overlap-tokens", "50", "--output-dir", "results/experiments/candidates20/$method"
        )
    }

    Invoke-PythonStage "Paper-cluster bootstrap statistics" @(
        "scripts/analyze_results.py", "--bootstrap-replicates", "2000"
    )
    Invoke-PythonStage "Development-set ablation summary" @("scripts/compute_ablations.py")
    Invoke-PythonStage "Held-out error analysis" @("scripts/analyze_errors.py", "--context-k", "5")
    Invoke-PythonStage "Regenerate report figures" @("scripts/generate_figures.py")

    if (-not $SkipDatasetPreparation) {
        Invoke-PythonStage "Prepare complete Qasper validation split" @(
            "scripts/prepare_dataset.py", "--seed", "42", "--dev-papers", "15",
            "--eval-papers", "281", "--output-dir", "data/processed/full_validation"
        )
    }
    foreach ($method in $methods) {
        Invoke-PythonStage "Full validation $method at K=5" @(
            "scripts/run_retrieval.py", "--split", "evaluation", "--method", $method,
            "--top-k", "5", "--chunking", "token-aware", "--max-tokens", "450",
            "--overlap-tokens", "50", "--data-dir", "data/processed/full_validation",
            "--output-dir", "results/experiments/full_validation/$method"
        )
    }
    foreach ($method in $methods) {
        Invoke-PythonStage "Full validation $method top-20 error-analysis candidates" @(
            "scripts/run_retrieval.py", "--split", "evaluation", "--method", $method,
            "--top-k", "20", "--chunking", "token-aware", "--max-tokens", "450",
            "--overlap-tokens", "50", "--data-dir", "data/processed/full_validation",
            "--output-dir", "results/experiments/full_validation/candidates20/$method"
        )
    }
    Invoke-PythonStage "Full-validation paper-cluster bootstrap statistics" @(
        "scripts/analyze_results.py", "--results-dir", "results/experiments/full_validation",
        "--output", "results/experiments/full_validation/statistical_summary.json",
        "--bootstrap-replicates", "2000", "--question-file",
        "data/processed/full_validation/evaluation_questions.jsonl"
    )
    Invoke-PythonStage "Full-validation figure generation" @(
        "scripts/generate_figures.py", "--summary-file",
        "results/experiments/full_validation/statistical_summary.json", "--output-dir",
        "docs/figures/full_validation", "--skip-ablations"
    )
    Invoke-PythonStage "Full-validation error analysis" @(
        "scripts/analyze_errors.py", "--prediction-root",
        "results/experiments/full_validation/candidates20", "--question-file",
        "data/processed/full_validation/evaluation_questions.jsonl", "--output",
        "results/experiments/full_validation/error_analysis.csv", "--context-k", "5"
    )
}
finally {
    Pop-Location
}
