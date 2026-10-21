param(
    [ValidateSet('all', 'cmrc2018', 'hotpotqa')]
    [string]$Dataset = 'all',
    [string]$Root = '',
    [ValidateSet('bm25', 'dense', 'hybrid', 'hybrid-rerank')]
    [string[]]$Methods = @('bm25', 'dense', 'hybrid', 'hybrid-rerank'),
    [int]$MaxQuestions = 0
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
$taskEmbedding = Join-Path $taskRoot 'models/bge-m3'
$taskReranker = Join-Path $taskRoot 'models/bge-reranker-base'
$taskRootData = if ($Root) { (Resolve-Path $Root -ErrorAction SilentlyContinue).Path } else { Join-Path $taskRoot '.test-tmp/retrieval-eval' }
if (-not $taskRootData) {
    New-Item -ItemType Directory -Force $Root | Out-Null
    $taskRootData = (Resolve-Path $Root).Path
}
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$taskOutput = Join-Path $taskRootData 'results'
if ($MaxQuestions -gt 0) {
    $taskOutput = Join-Path $taskRootData "check-$MaxQuestions"
}
$taskArgs = @(
    (Join-Path $PSScriptRoot 'evaluate_all.py'),
    '--root', $taskRootData,
    '--dataset', $Dataset,
    '--output-root', $taskOutput,
    '--embedding-model', $taskEmbedding,
    '--reranker-model', $taskReranker,
    '--embedding-device', 'cuda:0',
    '--reranker-device', 'cpu',
    '--embedding-batch-size', '1',
    '--reranker-batch-size', '1',
    '--methods'
) + $Methods
if ($MaxQuestions -gt 0) {
    $taskArgs += @('--max-questions', [string]$MaxQuestions)
}
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) {
    throw "Retrieval evaluation failed (exit $LASTEXITCODE)."
}
