@ECHO OFF
set SPHINXBUILD=..\VenvThesis\Scripts\sphinx-build.exe
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
