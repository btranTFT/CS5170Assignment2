from pathlib import Path


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def validate_pair(questions_path: Path, answers_path: Path) -> list[str]:
    errors: list[str] = []
    questions = read_nonempty_lines(questions_path)
    answers = read_nonempty_lines(answers_path)

    if len(questions) != len(answers):
        errors.append(
            f"Count mismatch: {questions_path} has {len(questions)} lines, "
            f"but {answers_path} has {len(answers)} lines."
        )

    for i, q in enumerate(questions, start=1):
        if not q.strip():
            errors.append(f"Empty question at {questions_path}:{i}.")

    for i, a in enumerate(answers, start=1):
        if not a.strip():
            errors.append(f"Empty answer at {answers_path}:{i}.")

    return errors


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    train_q = data_dir / "train" / "questions.txt"
    train_a = data_dir / "train" / "reference_answers.txt"
    test_q = data_dir / "test" / "questions.txt"
    test_a = data_dir / "test" / "reference_answers.txt"

    missing = [p for p in [train_q, train_a, test_q, test_a] if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"- {p}")
        raise SystemExit(1)

    errors: list[str] = []
    errors.extend(validate_pair(train_q, train_a))
    errors.extend(validate_pair(test_q, test_a))

    total_questions = len(read_nonempty_lines(train_q)) + len(read_nonempty_lines(test_q))
    if total_questions < 60 or total_questions > 80:
        errors.append(
            f"Total question count is {total_questions}; expected 60-80 for current project plan."
        )

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Validation passed.")
    print(f"Train size: {len(read_nonempty_lines(train_q))}")
    print(f"Test size: {len(read_nonempty_lines(test_q))}")
    print(f"Total size: {total_questions}")


if __name__ == "__main__":
    main()
