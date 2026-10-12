import os
import cairosvg
FIG = r"d:\cvs\pythoncode\AnomalyDetection\paper\figures"
SRC = os.path.join(FIG, "framework_src.svg")
PDF = os.path.join(FIG, "framework.pdf")
PNG = os.path.join(FIG, "framework_inspect.png")
cairosvg.svg2pdf(url=SRC, write_to=PDF)
cairosvg.svg2png(url=SRC, write_to=PNG, scale=2)
print("written:", PDF, os.path.getsize(PDF))
print("written:", PNG, os.path.getsize(PNG))
