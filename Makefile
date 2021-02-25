.phony = clean_all clean_stan clean_plots clean_pdf

BIBLIOGRAPHY = bibliography.bib
SAMPLES = $(shell find results/samples -name "*.csv")
LOGS = $(shell find results/samples -name "*.txt")
PLOTS = $(shell find results -name "*.png")
STAN_FILES =                      \
  model_kq_design_effects         \
  model_kq_design_effects.hpp     \
  model_kq_no_design_effects      \
  model_kq_no_design_effects.hpp
MARKDOWN_FILE = report.md
PDF_FILE = report.pdf
PANDOCFLAGS =                         \
  --from=markdown                     \
  --highlight-style=pygments          \
  --pdf-engine=xelatex                \
  --bibliography=$(BIBLIOGRAPHY)      

$(PDF_FILE): $(MARKDOWN_FILE) $(BIBLIOGRAPHY)
	pandoc $< -o $@ $(PANDOCFLAGS)

clean_all: clean_stan clean_plots clean_pdf

clean_stan:
	$(RM) $(SAMPLES) $(LOGS) $(STAN_FILES)

clean_plots:
	$(RM) $(PLOTS)

clean_pdf:
	$(RM) $(PDF_FILE)
