"""List training sessions with their billable token usage. Read-only; starts nothing.

  python -m forget7.sessions                         # today's sessions, newest first, with totals
  python -m forget7.sessions --since 17:00 --until 17:30   # a time window (local time, today)
  python -m forget7.sessions --all                   # every session the account has

There is no console page for training sessions; this is DescribeTrainingSession, which is
where the prefill / sample / training token counts live. Per-token prices are not published,
so this reports tokens, not money.
"""

import argparse
import datetime as dt

import boto3

from . import config as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="HH:MM local time today; default 00:00")
    ap.add_argument("--until", help="HH:MM local time today; default now")
    ap.add_argument("--all", action="store_true", help="no time filter")
    a = ap.parse_args()
    today = dt.datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    since = today + _hhmm(a.since) if a.since else today
    until = today + _hhmm(a.until) if a.until else dt.datetime.now().astimezone()

    sm = boto3.client("sagemaker", region_name=C.settings()["region"])
    sessions, token = [], None
    while True:
        resp = sm.list_training_sessions(MaxResults=100, **({"NextToken": token} if token else {}))
        sessions += resp.get("TrainingSessionSummaries", [])
        token = resp.get("NextToken")
        if not token:
            break
    if not a.all:
        sessions = [s for s in sessions if since <= s["CreationTime"] <= until]
    sessions.sort(key=lambda s: s["CreationTime"])

    tot = {"PrefillTokenCount": 0, "SampleTokenCount": 0, "TrainingTokenCount": 0}
    print(f"{'created':8} {'status':8} {'prefill':>9} {'sample':>9} {'train':>9}  name")
    for s in sessions:
        name = s["TrainingSessionArn"].rsplit("/", 1)[-1]
        d = sm.describe_training_session(TrainingSessionName=name)["TrainingSessionDetail"]
        use = d.get("BillableTokenUsage", {})
        for k in tot:
            tot[k] += use.get(k, 0)
        print(
            f"{s['CreationTime'].astimezone():%H:%M:%S} {d['TrainingSessionStatus']:8} "
            f"{use.get('PrefillTokenCount', 0):9,} {use.get('SampleTokenCount', 0):9,} "
            f"{use.get('TrainingTokenCount', 0):9,}  {name}"
        )
    print(
        f"{len(sessions)} sessions: prefill {tot['PrefillTokenCount']:,}, sample {tot['SampleTokenCount']:,}, "
        f"train {tot['TrainingTokenCount']:,} tokens"
    )


def _hhmm(s: str) -> dt.timedelta:
    h, m = s.split(":")
    return dt.timedelta(hours=int(h), minutes=int(m))


if __name__ == "__main__":
    main()
