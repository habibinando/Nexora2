param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GradleArgs)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$properties = Join-Path $root 'gradle/wrapper/gradle-wrapper.properties'
if (!(Test-Path $properties)) { throw "Missing Gradle wrapper properties: $properties" }
$line = Get-Content $properties | Where-Object { $_ -match '^distributionUrl=' } | Select-Object -First 1
if (!$line) { throw 'distributionUrl is missing from gradle/wrapper/gradle-wrapper.properties' }
$url = ($line -replace '^distributionUrl=', '') -replace '\\:', ':'
$shaLine = Get-Content $properties | Where-Object { $_ -match '^distributionSha256Sum=' } | Select-Object -First 1
if (!$shaLine) { throw 'A pinned distributionSha256Sum is required.' }
$expectedSha = ($shaLine -replace '^distributionSha256Sum=', '').ToLowerInvariant()
$file = [System.IO.Path]::GetFileName(([uri]$url).AbsolutePath)
$installName = ($file -replace '\.zip$', '') -replace '-bin$', ''
$userHome = if ($env:GRADLE_USER_HOME) { $env:GRADLE_USER_HOME } else { Join-Path $HOME '.gradle' }
$cache = Join-Path $userHome 'nexora-wrapper'
$install = Join-Path $cache $installName
$gradle = Join-Path $install 'bin/gradle.bat'
if (!(Test-Path $gradle)) {
  New-Item -ItemType Directory -Force -Path $cache | Out-Null
  $archive = Join-Path $cache ('.' + $file + '.' + [guid]::NewGuid().ToString('N'))
  $stage = Join-Path $cache ('.stage-' + [guid]::NewGuid().ToString('N'))
  Write-Host "Gradle bootstrap: downloading $url"
  Invoke-WebRequest -Uri $url -OutFile $archive
  $actualSha = (Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actualSha -ne $expectedSha) {
    Remove-Item $archive -Force
    throw "Gradle ZIP checksum mismatch: expected $expectedSha, received $actualSha"
  }
  Expand-Archive -Path $archive -DestinationPath $stage -Force
  Remove-Item $archive -Force
  $extracted = Join-Path $stage $installName
  if (!(Test-Path (Join-Path $extracted 'bin/gradle.bat'))) { throw "Unexpected Gradle distribution layout: $extracted" }
  if (Test-Path $install) { Remove-Item $install -Recurse -Force }
  Move-Item $extracted $install
  Remove-Item $stage -Recurse -Force
}
Push-Location $root
$code = 0
try {
  & $gradle @GradleArgs
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}
exit $code
