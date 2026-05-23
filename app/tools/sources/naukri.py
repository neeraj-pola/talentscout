# app/tools/sources/naukri.py
from app.tools.sources.base import HTTPProfileSource
from app.obs.events import log_event


class NaukriMockSource(HTTPProfileSource):
    source_name = "naukri"
    path_prefix = "/naukri"

    def search(self, queries, location=None, yoe_min=0, page=1, page_size=20):
        log_event(None, "tool.naukri", "search_start",
                  queries=queries, location=location, yoe_min=yoe_min, page=page)
        batch = super().search(queries, location, yoe_min, page, page_size)
        log_event(None, "tool.naukri", "search_end",
                  total_matched=batch.total_count, returned=len(batch.profiles),
                  next_page=batch.next_page)
        return batch

    def fetch_detail(self, source_id):
        log_event(None, "tool.naukri", "fetch_detail", source_id=source_id)
        return self._fetch_by_path(source_id)