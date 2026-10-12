$py = 'D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe'
$repo = 'd:\cvs\pythoncode\AnomalyDetection'
Write-Host "watchdog: waiting for GPU figure scripts (make_qual_heat / exp_multiseed) to finish..."
while ($true) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.CommandLine -like '*_make_qual_heat.py*' -or $_.CommandLine -like '*_exp_multiseed.py*'
    }
    if ($procs.Count -eq 0) { break }
    Start-Sleep -Seconds 15
}
Write-Host "watchdog: GPU scripts finished; running restyle_figs (error bars)..."
cd $repo\code\scripts
& $py _restyle_figs.py 2>&1 | Out-File -Append restyle.log
Write-Host "watchdog: compiling main.pdf..."
@('wpspdf.exe', 'SumatraPDF.exe', 'AcroRd32.exe') | ForEach-Object { taskkill /IM $_ /F 2>$null }
$env:PATH = $env:PATH + ';C:/Users/tiger/.workbuddy/binaries/tinytex/TinyTeX/bin/windows'
cd $repo\Paper
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null
bibtex main 2>&1 | Out-Null
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null
Write-Host "watchdog: git commit + tag v1.2 (local)..."
cd $repo
git add Paper/figures/fig_qual.pdf Paper/figures/fig_qual.png `
        Paper/figures/fig_heat.pdf Paper/figures/fig_heat.png `
        Paper/figures/fig_percat.pdf Paper/figures/fig_percat.png `
        Paper/figures/fig_cross.pdf Paper/figures/fig_cross.png `
        Paper/figures/data.json code/scripts/_exp_multiseed.py
git commit -m "GPU figures: PSC before/after (Raw row) for fig_qual/fig_heat; 3-seed error bars for Fig.3/7; finalize v1.2"
git tag v1.2
Write-Host "watchdog DONE: tag v1.2 created locally; push pending"
