#!/usr/bin/env bash
TMP=.griptemp.html
grip README.md --export $TMP
wkhtmltopdf $TMP README.pdf
rm $TMP
