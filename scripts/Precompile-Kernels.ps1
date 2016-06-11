
$scriptDir = $PSScriptRoot
$buildDir = $scriptDir + "/../build/"

$tensorSharpPath = Join-Path $buildDir "win_x64/TensorSharp/TensorSharp.dll"
$tsCudaPath = Join-Path $buildDir "win_x64/TensorSharp.CUDA/TensorSharp.CUDA.dll"

# Use LoadFrom instead of LoadFile - this way dependencies are loaded from the target
# path instead of powershell's directory
[Reflection.Assembly]::LoadFrom($tensorSharpPath)
[Reflection.Assembly]::LoadFrom($tsCudaPath)

[Environment]::CurrentDirectory = $buildDir

Write-Host "Beginning kernel precompile..."
[Console]::Out.Flush() 
$tsContext = new-object TensorSharp.CUDA.TSCudaContext
$tsContext.Precompile()
$tsContext.CleanUnusedPTX()
$tsContext.Dispose()
Write-Host "Done kernel precompile."
