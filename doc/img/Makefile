# Created by Andre Anjos <andre.anjos@idiap.ch> 
# Thu 22 Mar 2012 12:15:08 CET 

SVG := bob.svg bob-square.svg bob+reflection.svg overview.svg
PDF := $(SVG:.svg=.pdf)

all: $(PDF)

overview.pdf: overview.svg
	inkscape --export-pdf=$@ $<

%.pdf: %.svg
	inkscape --export-eps=$(shell basename $@ .pdf).eps $<
	epstopdf $(shell basename $@ .pdf).eps
	@rm -f $(shell basename $@ .pdf).eps

clean:
	@rm -f $(PDF)
	@rm -f *~

.PHONY: clean
