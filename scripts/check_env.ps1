<#
PowerShell helper that prints python executable and checks installed packages relevant to project.
Usage: .\.venv\Scripts\Activate.ps1; .\scripts\check_env.ps1
#>
Write-Host "Python:" (Get-Command python).Source
python -c "import sys,importlib; print('exe=', sys.executable); print('langchain=', importlib.util.find_spec('langchain') is not None); print('openai=', importlib.util.find_spec('openai') is not None); print('streamlit=', importlib.util.find_spec('streamlit') is not None)"
