"""Interactive terminal interface for the quantum diffusion POC.

Run this file, type a prompt containing a digit from 0 to 9, and it will
generate the corresponding image artifacts.
"""

from __future__ import annotations

from pathlib import Path

from quantum_diffusion_poc import (
    parse_prompt,
    generate_for_prompt,
    load_digit_data,
    save_metrics,
    save_visual_report,
    train_evaluator,
    build_digit_prototypes,
)
from shape_diffusion import (
    generate_shape_for_prompt,
    parse_shape_prompt,
    save_shape_metrics,
    save_shape_visual_report,
)


def detect_prompt_type(prompt: str) -> str:
    """Return 'digit' or 'shape' for supported prompts."""
    try:
        parse_prompt(prompt)
        return "digit"
    except ValueError:
        pass

    try:
        parse_shape_prompt(prompt)
        return "shape"
    except ValueError:
        pass

    raise ValueError(
        "I could not understand the prompt. Try 'generate digit 7' or "
        "'generate a triangle'."
    )


def main() -> None:
    print("Quantum-conditioned digit generator")
    print("Type a prompt such as: generate digit 7")
    print("Or try shapes/colors: yellow star, orange circle, blue triangle")
    print("Type q or quit to exit.\n")

    images, labels = load_digit_data()
    prototypes = build_digit_prototypes(images, labels)
    evaluator = train_evaluator(images, labels)

    output_dir = Path(__file__).resolve().parent / "outputs" / "terminal"
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        prompt = input("\nPrompt> ").strip()
        if prompt.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return

        if not prompt:
            print("Please enter a prompt containing a digit from 0 to 9.")
            continue

        try:
            prompt_type = detect_prompt_type(prompt)
        except ValueError as exc:
            print(exc)
            continue

        safe_prompt = "_".join(prompt.lower().split())
        if prompt_type == "digit":
            result = generate_for_prompt(
                prompt,
                prototypes,
                evaluator,
                shots=256,
                steps=20,
                seed=7,
            )
            report_path = output_dir / f"{safe_prompt}_report.png"
            metrics_path = output_dir / f"{safe_prompt}_metrics.json"
            save_visual_report([result], report_path)
            save_metrics([result], metrics_path)

            print("\nGenerated digit image.")
            print(f"Requested digit: {result.label}")
            print(f"Classical prediction: {result.classical_pred}")
            print(f"Quantum prediction: {result.quantum_pred}")
            print(f"Quantum MAE: {result.quantum_mae:.3f}")
            print(f"Image report: {report_path}")
            print(f"Metrics: {metrics_path}")
        else:
            result = generate_shape_for_prompt(prompt, shots=256, steps=20, seed=7)
            report_path = output_dir / f"{safe_prompt}_shape_report.png"
            metrics_path = output_dir / f"{safe_prompt}_shape_metrics.json"
            save_shape_visual_report(result, report_path)
            save_shape_metrics(result, metrics_path)

            print("\nGenerated shape image.")
            print(f"Requested shape: {result.shape}")
            print(f"Requested color: {result.color}")
            print(f"Quantum MAE: {result.quantum_mae:.3f}")
            print(f"Image report: {report_path}")
            print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
