@echo off
setlocal
set PROJECT_ROOT=%cd%
title jupyter @ mentorship_EPAM
call conda env create  -f environment.yml 
call conda activate mentorship_EPAM
call jupyter notebook