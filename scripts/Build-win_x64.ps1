
$scriptDir = $PSScriptRoot

nuget.exe restore ($scriptDir + "\..\src\TensorSharp.sln")


Import-Module -Name ($scriptDir + "\Invoke-MsBuild.psm1")

$buildResult = Invoke-MsBuild -Path ($scriptDir + "\..\src\TensorSharp.sln") -Params "/target:Clean;Build /property:Configuration=Release;Platform=x64" -ShowBuildOutputInNewWindow

if($buildResult.BuildSucceeded -eq $true)
{
	Write-Output "Build succeeded"
}
else
{
	Write-Output "Build failed: "
	Write-Output "Full command: "
	Write-Output $buildResult.CommandUsedToBuild 
	Write-Output "Message: "
	Write-Output $buildResult.Message
	
	
}


$tsOutputDir = $scriptDir + "\..\build\win_x64\TensorSharp"
md -Force $tsOutputDir
Copy-Item -path ($scriptDir + "\..\src\TensorSharp\bin\x64\Release\*.dll") -destination $tsOutputDir -Force
Copy-Item -path ($scriptDir + "\..\lib\win_x64\*") -destination $tsOutputDir -Force

$tsCudaOutputDir = $scriptDir + "\..\build\win_x64\TensorSharp.CUDA"
md -Force $tsCudaOutputDir
Copy-Item -path ($scriptDir + "\..\src\TensorSharp.CUDA\bin\x64\Release\*.dll") -destination $tsCudaOutputDir -Force
Remove-Item (Join-Path $tsCudaOutputDir "TensorSharp.dll")
