param(
    [ValidateSet('Release','Debug')][string]$Config = 'Release',
    [switch]$Clean,
    [switch]$CopyToPackage = $true,
    [string]$BuildDir = 'build'
)

$ErrorActionPreference = 'Stop'

function Resolve-RepoRoot {
    $here = Split-Path -Parent $MyInvocation.MyCommand.Path
    return (Resolve-Path (Join-Path $here '..')).Path
}

$repo = Resolve-RepoRoot
$buildPath = Join-Path $repo $BuildDir
$pkgPath = Join-Path $repo 'python\pylittle'

Write-Host "[INFO] repo=$repo"
Write-Host "[INFO] build=$buildPath"
Write-Host "[INFO] config=$Config"

if ($Clean) {
    if (Test-Path $buildPath) {
        Write-Host "[INFO] cleaning build dir"
        Remove-Item -Recurse -Force $buildPath
    }
}

if (-not (Test-Path $buildPath)) {
    New-Item -ItemType Directory -Path $buildPath | Out-Null
}

Push-Location $buildPath
try {
    Write-Host "[INFO] cmake configure"
    cmake -DPYLITTLE_BUILD_PYBIND=ON ..

    Write-Host "[INFO] cmake build"
    cmake --build . --config $Config
}
finally {
    Pop-Location
}

$pydGlob = Join-Path $buildPath ("bindings\\{0}\\_pylittle*.pyd" -f $Config)
$pyd = Get-ChildItem -Path $pydGlob -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $pyd) {
    throw "Cannot find built .pyd at: $pydGlob"
}

Write-Host "[INFO] built: $($pyd.FullName)"

if ($CopyToPackage) {
    if (-not (Test-Path $pkgPath)) {
        throw "Package path not found: $pkgPath"
    }

    $dst = Join-Path $pkgPath $pyd.Name
    Write-Host "[INFO] copying to: $dst"
    Copy-Item -Force $pyd.FullName $dst

    Write-Host "[INFO] import check"
    Push-Location $repo
    try {
        python -c "import pylittle._pylittle as m; print('OK import pylittle._pylittle:', m)"
    }
    finally {
        Pop-Location
    }
}

Write-Host "[OK] native build complete"