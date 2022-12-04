@echo off
setlocal
set PROJECT_ROOT=%cd%
set PYTHONPATH=%PYTHONPATH%;%cd%/src
title jupyter @ mentorship_EPAM
call conda activate mentorship_EPAM
call jupyter notebook