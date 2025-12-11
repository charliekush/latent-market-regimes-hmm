PDF_TARGET=report/main.tex

pdf:
	latexmk -g -pdf -bibtex -cd $(PDF_TARGET)

clean:
	latexmk -C -cd $(PDF_TARGET)
