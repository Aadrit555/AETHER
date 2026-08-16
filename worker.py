from __future__ import annotations

import time

from aether.core.database import init_db
from aether.services.jobs import worker

init_db()
worker.start()
while True:
    time.sleep(3600)
