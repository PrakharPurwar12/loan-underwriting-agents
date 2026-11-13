"""Small synchronous facade so scripts and tests can call the async system easily."""
import asyncio
from typing import Optional

from loan_uw.agents.underwriting_manager import UnderwritingManager
from loan_uw.models import ApplicantRecord
from loan_uw.runlog import AppendOnlyRunLog


def underwrite(text: str, runlog: Optional[AppendOnlyRunLog] = None,
               application_id: Optional[str] = None, records_path=None) -> ApplicantRecord:
    kwargs = {"records_path": records_path} if records_path else {}
    return asyncio.run(UnderwritingManager(runlog, **kwargs).process(text, application_id))
