$pdf_mode = 5;
$xelatex = 'xelatex -file-line-error -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';
# Overleaf import projects default to the project compiler setting; keep a XeLaTeX compatibility shim.
$pdflatex = 'xelatex -file-line-error -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';
$biber = 'biber %O %B';
$out_dir = 'build';
$aux_dir = 'build';
$max_repeat = 5;
