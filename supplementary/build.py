"""Compile the standalone report without modifying the companion manuscript."""
from pathlib import Path
import os
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
BUILD=HERE.parent/'build/supplement/latex'
BUILD.mkdir(parents=True,exist_ok=True)
latex=shutil.which('pdflatex')
if latex is None:
    local=Path.home()/'.TinyTeX/bin/universal-darwin/pdflatex'
    if local.exists():latex=str(local)
if latex is None:raise RuntimeError('Install TeX Live with pdflatex, or put pdflatex on PATH.')
cmd=[latex,'-no-shell-escape','-halt-on-error','-interaction=nonstopmode',f'-output-directory={BUILD}','main.tex']
for i in range(3):
    result=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True,errors='replace')
    (BUILD/f'pass-{i+1}.txt').write_text(result.stdout+result.stderr)
    if result.returncode:
        print(result.stdout[-6500:])
        raise RuntimeError(f'LaTeX pass {i+1} failed; see {BUILD}')
log=(BUILD/'main.log').read_text(errors='replace')
for problem in ['There were undefined references','There were undefined citations',
                'Overfull \\hbox','Overfull \\vbox','multiply defined']:
    if problem in log:raise RuntimeError(f'Layout/reference check failed: {problem}; see {BUILD}')
target=HERE/'supplementary_report.pdf'
shutil.copyfile(BUILD/'main.pdf',target)
print(target)
