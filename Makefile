.phony = clean_all clean_stan clean_plots clean_pdf clean_loo clean_ncdf

BIBLIOGRAPHY = bibliography.bib
SAMPLES = $(shell find results/samples -name "*.csv")
LOGS = $(shell find results/samples -name "*.txt")
PLOTS = $(shell find results/plots -name "*.svg")
STAN_FILES =                      \
  model_kq_design_effects         \
  model_kq_design_effects.hpp     \
  model_kq_no_design_effects      \
  model_kq_no_design_effects.hpp
LOO_FILES_PKL = $(shell find results/loo -name "*.pkl")
LOO_FILES_CSV = $(shell find results/loo -name "*.csv")
NCDF_FILES = $(shell find results/infd -name "*.ncdf")
STAN_INPUT_FILES = $(shell find results -name "*.json")
MARKDOWN_FILE = report.md
PDF_FILE = report.pdf
PANDOCFLAGS =                         \
  --from=markdown                     \
  --highlight-style=pygments          \
  --pdf-engine=xelatex                \
  --bibliography=$(BIBLIOGRAPHY)      

$(PDF_FILE): $(MARKDOWN_FILE) $(BIBLIOGRAPHY)
	pandoc $< -o $@ $(PANDOCFLAGS)

clean_all: clean_stan clean_plots clean_pdf clean_samples clean_loo clean_ncdf

clean_stan:
	$(RM) $(SAMPLES) $(LOGS) $(STAN_FILES) $(STAN_INPUT_FILES)

clean_plots:
	$(RM) $(PLOTS)

clean_pdf:
	$(RM) $(PDF_FILE)

clean_loo:
	$(RM) $(LOO_FILES_PKL) $(LOO_FILES_CSV)

clean_ncdf:
	$(RM) $(NCDF_FILES)
