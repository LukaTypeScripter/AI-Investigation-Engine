# INC-2207 Payment processing delay (5 March 2025)

Severity: SEV-2. Status: resolved.

## Summary
Between 09:12 and 12:22 UTC on 5 March 2025 payment authorisations were delayed. The total duration was 3 hours 10 minutes. About 12,400 transactions were authorised late.

## Root cause
Change CHG-1041 deployed payment-gateway v4.12 on 4 March and reduced the database connection pool from 200 to 50. During the morning peak on 5 March requests queued for connections and authorisation latency rose.

## Impact
- payment-gateway: p99 authorisation latency rose from 400 ms to 38 seconds.
- ledger-service: received delayed bursts of writes and built up a write backlog.
- notification-service: payment receipts were delayed by up to 2 hours.

## Resolution
The gateway was rolled back to v4.11 at 12:15 UTC on 5 March 2025 (change CHG-1044), which restored the connection pool size of 200.

## Follow-up
See postmortem-INC-2207.html for the action items. Related incidents: INC-2211 (ledger write backlog) and INC-2219 (delayed receipt emails).
