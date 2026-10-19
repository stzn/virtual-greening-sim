from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SimClock:
    """仮想時計。現実の時刻には依存しない。"""

    now: datetime

    def advance(self, hours: int = 1) -> None:
        self.now += timedelta(hours=hours)
