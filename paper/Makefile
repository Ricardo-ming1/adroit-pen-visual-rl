OVERLEAF_IMAGE ?= local/adroit-overleaf-ce:6.3.0
REPORT_NAME := adroit_pen_technical_report_zh

.PHONY: pdf docker-pdf zip figures clean

pdf:
	mkdir -p build dist
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf dist/$(REPORT_NAME).pdf

docker-pdf:
	mkdir -p build dist
	docker run --rm --entrypoint /bin/bash --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$$(pwd):/work" -w /work $(OVERLEAF_IMAGE) \
		-lc 'latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex && cp build/main.pdf dist/$(REPORT_NAME).pdf'

zip:
	bash scripts/build_overleaf_zip.sh

figures:
	bash scripts/rebuild_figures.sh

clean:
	latexmk -C -outdir=build main.tex || true
	rm -f build/* dist/$(REPORT_NAME).pdf dist/adroit_pen_overleaf.zip
