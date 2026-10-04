"""Math Router for MathSolver AI."""

import base64
import json
import re
import urllib.error
import urllib.request

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TEXT_MODEL = "qwen2.5:7b"
VISION_MODEL = "qwen2.5vl:7b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

SYSTEM_PROMPT = """
You are MathSolver AI, an expert, patient mathematics teacher.

Solve arithmetic, fractions, decimals, percentages, ratios, algebra,
equations, inequalities, exponents, roots, logarithms, sequences,
geometry, circles, angles, perimeter, area, volume, coordinate geometry,
graphs, trigonometry, calculus, vectors, matrices, statistics, probability,
proofs, and word problems.

TEACHING RULES:
1. Start directly with the solution.
2. Use clear, student-friendly English.
3. Show formulas and explain important steps.
4. Substitute known values before calculating.
5. Keep units in measurement questions.
6. Never invent missing measurements or image details.
7. Check answers when practical.
8. Answer each part of a multi-part question separately.
9. Use readable mathematical notation, not raw LaTeX commands.
10. Finish with Final Answer: and state the answer on the next line.
"""


def clean_image(image_data):
    """Validate base64 image data."""
    if not image_data:
        return None

    image_data = image_data.strip()

    if image_data.startswith("data:"):
        if "," not in image_data:
            raise ValueError("Invalid image data URL.")
        image_data = image_data.split(",", 1)[1]

    try:
        base64.b64decode(image_data, validate=True)
    except Exception as exc:
        raise ValueError("Invalid image data. Please upload it again.") from exc

    return image_data


def call_ollama(model, messages):
    """Send a request to Ollama."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result.get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer.")
        return answer

    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from exc


def clean_math_output(answer):
    """Remove common LaTeX and Markdown formatting."""
    answer = answer.replace("\\[", "").replace("\\]", "")
    answer = answer.replace("\\(", "").replace("\\)", "")
    answer = answer.replace("$$", "").replace("$", "")

    pattern = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(10):
        updated = pattern.sub(r"(\1)/(\2)", answer)
        if updated == answer:
            break
        answer = updated

    answer = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", answer)
    answer = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer, flags=re.DOTALL)
    answer = re.sub(r"(?m)^\s*#{1,6}\s*", "", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def verify_linear_equation(question):
    """Solve a supported, simple linear equation with SymPy."""
    if not isinstance(question, str):
        return None

    text = question.strip().replace("−", "-").replace("×", "*")

    if len(text) > 160 or text.count("=") != 1:
        return None

    if not re.fullmatch(r"[0-9a-zA-Z+\-*/^().\s=]+", text):
        return None

    left_text, right_text = [part.strip() for part in text.split("=")]

    if not left_text or not right_text:
        return None

    variables = set(re.findall(r"[a-zA-Z]", text))
    if len(variables) != 1:
        return None

    variable_name = next(iter(variables))
    variable = sp.Symbol(variable_name)

    try:
        local_symbols = {variable_name: variable}

        left = parse_expr(
            left_text,
            local_dict=local_symbols,
            transformations=TRANSFORMATIONS,
        )
        right = parse_expr(
            right_text,
            local_dict=local_symbols,
            transformations=TRANSFORMATIONS,
        )

        # Reject nonlinear equations in this first version.
        difference = sp.expand(left - right)
        if sp.degree(difference, variable) != 1:
            return None

        solutions = sp.solve(sp.Eq(left, right), variable)
        if not solutions:
            return None

        verified = [
            str(value)
            for value in solutions
            if sp.simplify(left.subs(variable, value)
                           - right.subs(variable, value)) == 0
        ]

        if not verified:
            return None

        return {
            "method": "SymPy",
            "variable": variable_name,
            "solutions": verified,
            "verified": True,
        }

    except Exception:
        return None


def route_math(question, image_base64=None, history=None):
    """Route text and image questions through Ollama."""
    question = (question or "").strip()
    image = clean_image(image_base64) if image_base64 else None

    if not question and not image:
        raise ValueError("Enter a question or upload a math image.")

    prompt = (
        "Solve the student's mathematics question step by step.\n"
        f"Student's exact request: {question or 'Read and solve the uploaded image.'}\n"
        "If an image is attached, inspect it and locate the exact requested "
        "question number and subpart. Read the visible labels, measurements, "
        "and instructions. If the requested part is readable, solve it now "
        "with calculations instead of asking the student to describe it. "
        "If essential information is genuinely unreadable, identify exactly "
        "what cannot be read. Never invent missing values."
    )

    if question and not image:
        result = verify_linear_equation(question)
        if result:
            prompt += (
                "\nIndependent SymPy calculation:\n"
                f"Variable: {result['variable']}\n"
                f"Solution(s): {', '.join(result['solutions'])}\n"
                "Substitution verification passed.\n"
                "Explain the steps and ensure they match the question."
            )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if isinstance(history, list):
        for item in history[-8:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})

    user_message = {"role": "user", "content": prompt}
    if image:
        user_message["images"] = [image]

    messages.append(user_message)

    model = VISION_MODEL if image else TEXT_MODEL
    return clean_math_output(call_ollama(model, messages))


def solve_math(question, image_base64=None, history=None):
    """Compatibility function for the Flask application."""
    return route_math(question, image_base64, history)
