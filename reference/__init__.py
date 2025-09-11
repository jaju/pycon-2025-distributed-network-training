"""Reference (production-leaning) training scripts.

Contains the richer, instrumented variants of the workshop code:
- simple_train.py         (baseline)
- simple_dist_train.py    (manual sync)
- ddp_train.py            (DDP)
- fsdp_train.py           (FSDP)

Config and metrics remain under stepN.* during the transition and will be
unified here in a subsequent phase.
"""
