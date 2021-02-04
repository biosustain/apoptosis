.phony = clean_all clean_stan clean_plots

SAMPLES = $(shell find results/samples -name "*.csv")
LOGS = $(shell find results/samples -name "*.txt")
PLOTS = $(shell find results -name "*.png")
STAN_FILES = model model.hpp validation_model validation_model.hpp
MARKDOWN_FILE = report.md
PDF_FILE = report.pdf
PANDOCFLAGS =                         \
  --from=markdown                     \
  --highlight-style=pygments          \
  --pdf-engine=xelatex

$(PDF_FILE): $(MARKDOWN_FILE)
	pandoc $< -o $@ $(PANDOCFLAGS)

clean_all: clean_stan clean_plots

clean_stan:
	$(RM) $(SAMPLES) $(LOGS) $(STAN_FILES)

clean_plots:
	$(RM) $(PLOTS)
