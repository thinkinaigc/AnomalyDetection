$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($p in $procs) {
    $cmd = $p.CommandLine
    if ($null -eq $cmd) { $cmd = '<none>' }
    Write-Output ("{0}|{1}" -f $p.ProcessId, $cmd)
}
