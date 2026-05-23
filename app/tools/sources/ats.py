# app/tools/sources/ats.py
from app.tools.sources.base import HTTPProfileSource
from app.obs.events import log_event


class ATSMockSource(HTTPProfileSource):
    source_name = "ats"
    path_prefix = "/ats"

    def search(self, queries, location=None, yoe_min=0, page=1, page_size=20):
        log_event(None, "tool.ats", "search_start",
                  queries=queries, location=location, yoe_min=yoe_min, page=page)
        batch = super().search(queries, location, yoe_min, page, page_size)
        log_event(None, "tool.ats", "search_end",
                  total_matched=batch.total_count, returned=len(batch.profiles),
                  next_page=batch.next_page)
        return batch

    def fetch_detail(self, source_id):
        log_event(None, "tool.ats", "fetch_detail", source_id=source_id)
        return self._fetch_by_path(source_id)