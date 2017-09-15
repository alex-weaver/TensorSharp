# To build the nuget:
#	.\Build-win_x64.ps1 -Configuration ReleaseCpuOnly -BuildNuget

param(
	[parameter(Mandatory=$false,HelpMessage="The build configuration [Release/Debug/ReleaseCpuOnly].")]
	[ValidateNotNullOrEmpty()]
	[string] $Configuration = "Release",

	[parameter(Mandatory=$false,HelpMessage="Build the nuget package.")]
	[ValidateNotNullOrEmpty()]
	[switch] $BuildNuget = $false
)

$scriptDir = $PSScriptRoot

nuget.exe restore ($scriptDir + "\..\src\TensorSharp.sln")

Import-Module -Name ($scriptDir + "\Invoke-MsBuild.psm1")

$buildResult = Invoke-MsBuild -Path ($scriptDir + "\..\src\TensorSharp.sln") -Params ("/target:Clean;Build /property:Configuration=" + $Configuration + ";Platform=x64") -ShowBuildOutputInNewWindow

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
Copy-Item -path ($scriptDir + "\..\src\TensorSharp\bin\x64\" + $Configuration + "\*.dll") -destination $tsOutputDir -Force
Copy-Item -path ($scriptDir + "\..\src\CpuOps\bin\x64\" + $Configuration + "\*.dll") -destination $tsOutputDir -Force
Copy-Item -path ($scriptDir + "\..\lib\win_x64\*") -destination $tsOutputDir -Force

if($Configuration -eq "ReleaseCpuOnly")
{
	if($BuildNuget)
	{
		$nugetDir = $scriptDir + "\..\nuget\"
	
		New-Item ($nugetDir + "lib\net45") -Force
		New-Item ($nugetDir + "build\x64") -Force
		New-Item ($nugetDir + "build\x64") -Force
	
		Copy-Item -path ($scriptDir + "\..\src\TensorSharp\bin\x64\" + $Configuration + "\*.dll") -destination ($nugetDir + "lib\net45") -Force
		Copy-Item -path ($scriptDir + "\..\src\CpuOps\bin\x64\" + $Configuration + "\*.dll") -destination ($nugetDir + "build\x64") -Force
		Copy-Item -path ($scriptDir + "\..\lib\win_x64\*") -destination ($nugetDir + "build\x64") -Force
	
		$location = Get-Location
		Set-Location $nugetDir
	
		nuget.exe pack .\TensorSharp.nuspec
	
		Set-Location $location
	}
}
else
{
	$tsCudaOutputDir = $scriptDir + "\..\build\win_x64\TensorSharp.CUDA"
	md -Force $tsCudaOutputDir
	Copy-Item -path ($scriptDir + "\..\src\TensorSharp.CUDA\bin\x64\Release\*.dll") -destination $tsCudaOutputDir -Force
	Remove-Item (Join-Path $tsCudaOutputDir "TensorSharp.dll")
}