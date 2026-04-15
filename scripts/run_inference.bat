@echo off
setlocal

REM Usage:
REM scripts\run_inference.bat data\test\questions.txt system_outputs\system_output_1.txt artifacts\run_logs\output_1_log.json

if "%~3"=="" (
  echo Usage: scripts\run_inference.bat ^<questions_file^> ^<output_file^> ^<retrieval_log_file^>
  exit /b 1
)

python -m src.rag.answer_questions --questions-file "%~1" --output-file "%~2" --retrieval-log-file "%~3"
