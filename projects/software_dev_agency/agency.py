from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


EventCallback = Callable[["AgentEvent"], None]


@dataclass
class AgentEvent:
    agent: str
    event_type: str
    message: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


@dataclass
class AgencyRun:
    run_id: str
    request: str
    mode: str
    status: str = "queued"
    specification: str = ""
    implementation: str = ""
    test_report: str = ""
    qa_passed: bool = False
    revision_count: int = 0
    events: list[AgentEvent] = field(default_factory=list)
    artifact_dir: str = ""


class SoftwareDevAgency:
    """Coordinates a PM -> Coder -> QA loop without executing generated code."""

    def __init__(
        self,
        output_dir: Path | None = None,
        mode: str = "Demo",
        on_event: EventCallback | None = None,
    ) -> None:
        self.mode = mode
        self.output_dir = output_dir or Path(__file__).resolve().parent / "runs"
        self.on_event = on_event

    def run(self, request: str, max_revisions: int = 2) -> AgencyRun:
        request = request.strip()
        if not request:
            raise ValueError("Describe the software task before starting a run.")
        if max_revisions < 0 or max_revisions > 3:
            raise ValueError("max_revisions must be between 0 and 3.")

        run = AgencyRun(
            run_id=uuid.uuid4().hex[:8],
            request=request,
            mode=self.mode,
        )
        self._emit(run, "system", "run_started", "Agency run started.")
        run.status = "in_progress"

        run.specification = self._pm(run, request)
        run.implementation = self._coder(run, run.specification)

        for revision in range(max_revisions + 1):
            run.revision_count = revision
            run.test_report, run.qa_passed = self._qa(
                run,
                run.specification, run.implementation
            )
            if run.qa_passed:
                break
            if revision == max_revisions:
                break
            self._emit(
                run,
                "Coder",
                "revision_requested",
                f"QA requested revision {revision + 1}.",
            )
            run.implementation = self._coder(
                run,
                f"{run.specification}\n\nQA feedback:\n{run.test_report}"
            )

        run.status = "completed" if run.qa_passed else "needs_review"
        self._persist(run)
        self._emit(
            run,
            "system",
            "run_completed",
            "Run completed with a passing QA gate."
            if run.qa_passed
            else "Run completed and needs human review.",
        )
        self._persist(run)
        return run

    def _pm(self, run: AgencyRun, request: str) -> str:
        self._emit(run, "PM", "started", "Turning the request into an executable brief.")
        if self.mode == "Demo":
            result = (
                f"# Product brief\n\n## Goal\n{request}\n\n"
                "## Acceptance criteria\n"
                "- The primary user flow is clear and usable.\n"
                "- Invalid or empty input is handled visibly.\n"
                "- The result is testable without external services.\n\n"
                "## Deliverables\n- A small implementation\n- Focused tests\n- A short usage note"
            )
        else:
            result = self._ask_llm(
                "You are the PM Agent. Convert this request into a concise product brief "
                "with goal, acceptance criteria, edge cases, and deliverables:\n\n" + request
            )
        self._emit(run, "PM", "completed", "Product brief ready.")
        return result

    def _coder(self, run: AgencyRun, specification: str) -> str:
        self._emit(run, "Coder", "started", "Implementing the product brief.")
        if self.mode == "Demo":
            result = (
                "```python\n"
                "def build_solution(user_input: str) -> str:\n"
                '    if not user_input.strip():\n'
                '        raise ValueError("Input is required")\n'
                '    return f"Processed: {user_input.strip()}"\n'
                "```\n\n"
                "```python\n"
                "def test_build_solution():\n"
                '    assert build_solution("demo") == "Processed: demo"\n'
                "```\n"
            )
        else:
            result = self._ask_llm(
                "You are the Coder Agent. Produce a minimal implementation and focused "
                "tests from this brief. Return code blocks only, with no claims that "
                "you executed them:\n\n" + specification
            )
        self._emit(run, "Coder", "completed", "Implementation and tests ready for QA.")
        return result

    def _qa(
        self, run: AgencyRun, specification: str, implementation: str
    ) -> tuple[str, bool]:
        self._emit(run, "QA", "started", "Reviewing acceptance criteria and implementation.")
        if self.mode == "Demo":
            passed = bool(
                re.search(r"def build_solution", implementation)
                and re.search(r"ValueError", implementation)
                and re.search(r"def test_", implementation)
            )
            report = (
                "PASS — implementation includes the main function, empty-input "
                "validation, and a focused test."
                if passed
                else "FAIL — expected implementation and test markers were not found."
            )
        else:
            report = self._ask_llm(
                "You are the QA Agent. Review the brief and implementation. Return a "
                "short report beginning with PASS or FAIL, followed by concrete "
                "findings. Do not claim to execute code:\n\nBRIEF:\n"
                + specification
                + "\n\nIMPLEMENTATION:\n"
                + implementation
            )
            passed = report.lstrip().upper().startswith("PASS")
        self._emit(
            run,
            "QA",
            "passed" if passed else "failed",
            "QA gate passed." if passed else "QA found issues.",
        )
        return report, passed

    def _ask_llm(self, prompt: str) -> str:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI mode. Switch to Demo mode or configure the key."
            )
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("AGENCY_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are one member of a disciplined software agency."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The agent returned an empty response.")
        return content.strip()

    def _emit(
        self,
        run: AgencyRun | None,
        agent: str,
        event_type: str,
        message: str,
    ) -> None:
        event = AgentEvent(agent=agent, event_type=event_type, message=message)
        if run is not None:
            run.events.append(event)
        if self.on_event:
            self.on_event(event)

    def _persist(self, run: AgencyRun) -> None:
        artifact_dir = self.output_dir / run.run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run.artifact_dir = str(artifact_dir)
        (artifact_dir / "product_brief.md").write_text(run.specification, encoding="utf-8")
        (artifact_dir / "implementation.md").write_text(run.implementation, encoding="utf-8")
        (artifact_dir / "qa_report.md").write_text(run.test_report, encoding="utf-8")
        (artifact_dir / "run.json").write_text(
            json.dumps(asdict(run), indent=2), encoding="utf-8"
        )
