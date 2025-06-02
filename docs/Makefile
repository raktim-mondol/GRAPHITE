# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for GRAPHITE documentation
clean:
	@echo "Removing build directory..."
	@rm -rf $(BUILDDIR)

html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

livehtml:
	@echo "Starting live HTML build with auto-reload..."
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

linkcheck:
	@echo "Checking external links..."
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

doctest:
	@echo "Running doctests..."
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS)

coverage:
	@echo "Checking documentation coverage..."
	@$(SPHINXBUILD) -b coverage "$(SOURCEDIR)" "$(BUILDDIR)/coverage" $(SPHINXOPTS)

pdf:
	@echo "Building PDF documentation..."
	@$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	@make -C $(BUILDDIR)/latex all-pdf

install:
	@echo "Installing documentation dependencies..."
	@pip install -r requirements.txt

serve:
	@echo "Serving documentation on http://localhost:8000"
	@cd $(BUILDDIR)/html && python -m http.server 8000

# Advanced targets
autodoc:
	@echo "Regenerating API documentation..."
	@sphinx-apidoc -f -o api ../training_step_1/mil_classification/src
	@sphinx-apidoc -f -o api ../training_step_2/self_supervised_training
	@sphinx-apidoc -f -o api ../visualization_step_1/xai_visualization/src
	@sphinx-apidoc -f -o api ../visualization_step_2/fusion_visualization

full: clean autodoc html
	@echo "Full documentation build completed!"

# Development targets
dev: install html serve

# Help target with project-specific information
graphite-help:
	@echo "GRAPHITE Documentation Build System"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  html         Build HTML documentation"
	@echo "  clean        Remove build directory"
	@echo "  livehtml     Start live HTML build with auto-reload"
	@echo "  linkcheck    Check external links"
	@echo "  doctest      Run doctests"
	@echo "  coverage     Check documentation coverage"
	@echo "  pdf          Build PDF documentation"
	@echo "  install      Install documentation dependencies"
	@echo "  serve        Serve documentation on localhost:8000"
	@echo "  autodoc      Regenerate API documentation"
	@echo "  full         Clean + autodoc + html"
	@echo "  dev          Install + html + serve"
	@echo ""
	@echo "Examples:"
	@echo "  make html                    # Build HTML docs"
	@echo "  make clean html              # Clean build and rebuild"
	@echo "  make dev                     # Full development setup"
	@echo "  make SPHINXOPTS='-W' html    # Build with warnings as errors"
