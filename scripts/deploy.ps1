<#
.SYNOPSIS
    部署腳本
.PARAMETER ImageTag
    Docker 映像檔的 Tag 或 Commit Hash
.EXAMPLE
    .\scripts\deploy.ps1 4864a00220fe66ad21320a2fb7aa9aes3ee6b306e
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ImageTag 
)

$ErrorActionPreference = "Stop"

$Deployment = "mlops-demo-api"
$Container = "mlops-demo-api"
$Image = "ghcr.io/yiyi2525/mlops-demo:$ImageTag"

Write-Host "=== MLOps Demo Deployment ==="

# 1. Check Kubernetes context
$context = kubectl config current-context
Write-Host "Current Kubernetes context: $context"

# 2. Check cluster connectivity
Write-Host "`nChecking Kubernetes cluster..."
kubectl get nodes

# 3. Show the image to be deployed
Write-Host "`nImage to deploy:"
Write-Host $Image

$confirm = Read-Host "Continue deployment? (y/n)"

if ($confirm -ne "y") {
    Write-Host "Deployment cancelled."
    exit 0
}

# 4. Update Deployment image
Write-Host "`nUpdating Deployment..."
kubectl set image deployment/$Deployment `
    $Container=$Image

# 5. Wait for Rolling Update
Write-Host "`nWaiting for rollout..."
kubectl rollout status deployment/$Deployment --timeout=180s

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nRollout failed. Rolling back..."

    kubectl rollout undo deployment/$Deployment
    kubectl rollout status deployment/$Deployment --timeout=180s

    throw "Deployment failed and rollback was triggered."
}

# 6. Verify result
Write-Host "`nDeployment completed successfully."

kubectl get pods

Write-Host "`nRunning images:"
kubectl get pods `
    -o jsonpath="{range .items[*]}{.metadata.name}{'  '}{.spec.containers[0].image}{'\n'}{end}"
