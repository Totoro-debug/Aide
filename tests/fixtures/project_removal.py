"""Bounded observation of the production Project removal operation API."""

import asyncio

from myclaw.service.runtime import LocalService


async def wait_for_project_removal(
    service: LocalService, client_id: str, project_id: str, operation_id: str
) -> dict[str, object]:
    async with asyncio.timeout(10):
        while True:
            status = await service.project_removal_status(client_id, project_id, operation_id)
            if status["status"] in {"completed", "failed"}:
                return status
            await asyncio.sleep(0.01)


async def complete_project_removal(
    service: LocalService, client_id: str, project_id: str
) -> dict[str, object]:
    started = await service.start_project_removal(client_id, project_id)
    operation_id = started["operation_id"]
    assert isinstance(operation_id, str)
    status = await wait_for_project_removal(service, client_id, project_id, operation_id)
    assert status["status"] == "completed"
    return status
