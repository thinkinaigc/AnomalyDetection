Add-Type -AssemblyName System.Drawing
$files = @(
  'd:\cvs\pythoncode\AnomalyDetection\paper\figures\fig_qual.png',
  'd:\cvs\pythoncode\AnomalyDetection\paper\figures\fig_heat.png'
)
foreach ($f in $files) {
    $img = [System.Drawing.Image]::FromFile($f)
    $w = $img.Width; $h = $img.Height
    Write-Output ("{0}: {1}x{2}px  = {3:N2} x {4:N2} in @900dpi" -f (Split-Path $f -Leaf), $w, $h, ($w/900), ($h/900))
    $img.Dispose()
}
