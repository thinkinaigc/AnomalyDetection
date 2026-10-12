$py = 'D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe'
$repo = 'd:\cvs\pythoncode\AnomalyDetection'
function log($m){ Write-Output $m }

# 1) fig_qual / fig_heat (4-row: Input / GT / Raw before PSC / Ours after PSC)
cd $repo\code\scripts
log "=== [1/4] make_qual_heat (fig_qual/fig_heat) ==="
& $py _make_qual_heat.py 2>&1 | Tee-Object -FilePath qual_heat2.log
if (-not (Test-Path $repo\Paper\figures\fig_qual.pdf)) { log "ERROR: fig_qual.pdf not produced"; exit 1 }

# 2) error bars (fig_percat / fig_cross) using written std
log "=== [2/4] restyle_figs (error bars) ==="
& $py _restyle_figs.py 2>&1 | Tee-Object -FilePath restyle2.log

# 3) compile main.pdf (with updated captions)
log "=== [3/4] compile main.pdf ==="
@('wpspdf.exe', 'SumatraPDF.exe', 'AcroRd32.exe') | ForEach-Object { taskkill /IM $_ /F 2>$null }
$env:PATH = $env:PATH + ';C:/Users/tiger/.workbuddy/binaries/tinytex/TinyTeX/bin/windows'
cd $repo\Paper
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null
bibtex main 2>&1 | Out-Null
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null
pdflatex -interaction=nonstopmode main.tex 2>&1 | Out-Null

# 4) commit + tag v1.2
log "=== [4/4] git commit + tag v1.2 ==="
cd $repo
git add Paper/figures/fig_qual.pdf Paper/figures/fig_qual.png `
        Paper/figures/fig_heat.pdf Paper/figures/fig_heat.png `
        Paper/figures/fig_percat.pdf Paper/figures/fig_percat.png `
        Paper/figures/fig_cross.pdf Paper/figures/fig_cross.png `
        Paper/figures/data.json `
        code/scripts/_make_qual_heat.py code/scripts/_exp_multiseed.py `
        Paper/sections/04_experiments.tex
git commit -m "GPU figures v1.2: fig_qual/fig_heat PSC before/after (Raw row); Fig.3/7 error bars (3-seed std); captions updated"
git tag -f v1.2
log "=== DONE: v1.2 created locally; push pending ==="
