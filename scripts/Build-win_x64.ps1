
$scriptDir = $PSScriptRoot

$tsOutputDir = $scriptDir + "\..\build\win_x64\TensorSharp"
md -Force $tsOutputDir
Copy-Item -path ($scriptDir + "\..\src\TensorSharp\bin\x64\Release\*.dll") -destination $tsOutputDir -Force
Copy-Item -path ($scriptDir + "\..\lib\win_x64\*") -destination $tsOutputDir -Force

$tsCudaOutputDir = $scriptDir + "\..\build\win_x64\TensorSharp.CUDA"
md -Force $tsCudaOutputDir
Copy-Item -path ($scriptDir + "\..\src\TensorSharp.CUDA\bin\x64\Release\*.dll") -destination $tsCudaOutputDir -Force
Remove-Item (Join-Path $tsCudaOutputDir "TensorSharp.dll")
