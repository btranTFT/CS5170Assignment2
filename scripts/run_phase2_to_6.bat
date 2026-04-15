@echo off
setlocal

REM Run from repository root.
set PY=c:/python314/python.exe

%PY% scripts/validate_phase2_data.py || exit /b 1
%PY% -m src.rag.build_index --data-dir data --index-dir artifacts/retrieval_index || exit /b 1

%PY% -m src.rag.answer_questions --questions-file data/test/questions.txt --output-file system_outputs/system_output_1.txt --retrieval-log-file artifacts/run_logs/output_1_log.json --index-dir artifacts/retrieval_index --top-k 5 --no-reader || exit /b 1
%PY% -m src.rag.evaluate --predictions system_outputs/system_output_1.txt --references data/test/reference_answers.txt --output-json artifacts/metrics/system_output_1_metrics.json || exit /b 1

%PY% -m src.rag.answer_questions --questions-file data/test/questions.txt --output-file system_outputs/system_output_2.txt --retrieval-log-file artifacts/run_logs/output_2_log.json --index-dir artifacts/retrieval_index --top-k 8 --no-reader || exit /b 1
%PY% -m src.rag.evaluate --predictions system_outputs/system_output_2.txt --references data/test/reference_answers.txt --output-json artifacts/metrics/system_output_2_metrics.json || exit /b 1

%PY% -m src.rag.answer_questions --questions-file data/test/questions.txt --output-file system_outputs/system_output_3.txt --retrieval-log-file artifacts/run_logs/output_3_log.json --index-dir artifacts/retrieval_index --top-k 5 --no-reader || exit /b 1
%PY% -m src.rag.evaluate --predictions system_outputs/system_output_3.txt --references data/test/reference_answers.txt --output-json artifacts/metrics/system_output_3_metrics.json || exit /b 1

%PY% -m src.rag.significance --pred-a system_outputs/system_output_2.txt --pred-b system_outputs/system_output_1.txt --references data/test/reference_answers.txt --output-json artifacts/metrics/significance_2_vs_1.json || exit /b 1

echo Phase 2 to 6 pipeline completed.
