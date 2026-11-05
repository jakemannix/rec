"""Experiment-only expert-team extension; does not change supported devices."""

from rec_mlx.kernels.kv_teams import function


def validate_team_tiles(mlp_workers: int) -> None:
    """Check pair/quad row ownership for six routed and one shared expert."""
    if mlp_workers not in (32, 40, 48, 56, 64):
        raise ValueError("Experimental MLP workers must be 32, 40, 48, 56, or 64")
    team = mlp_workers // 8
    for shared in (False, True):
        groups = team * (2 if shared else 1)
        for width, rows_per_simd in ((1024 if shared else 512, 2), (2048, 4)):
            rows = []
            for group in range(groups):
                for simd in range(8):
                    for first in range(
                        (group * 8 + simd) * rows_per_simd, width, groups * 8 * rows_per_simd
                    ):
                        rows.extend(range(first, first + rows_per_simd))
            if sorted(rows) != list(range(width)):
                raise AssertionError(
                    f"Invalid row coverage for {mlp_workers=}, {shared=}, {width=}"
                )


def expert_teams_transform(header: str, mlp_workers: int) -> str:
    """Apply production team transformations with audited intermediate counts.

    Intermediate counts alter work balance, not each output's arithmetic order.
    Complete shader parity and residency still require hardware validation.
    """
    validate_team_tiles(mlp_workers)
    team = mlp_workers // 8
    for name in ("expert_team_gateup", "expert_team_down", "expert_team_handoff"):
        old = function(header, name)
        new = old.replace("group>=24", f"group>={team * 6}")
        new = new.replace("group-24", f"group-{team * 6}")
        new = new.replace("group/4", f"group/{team}").replace("group%4", f"group%{team}")
        new = new.replace("shared?8:4", f"shared?{team * 2}:{team}")
        new = new.replace("?24:", f"?{team * 6}:")
        new = new.replace(")*4, count=", f")*{team}, count=")
        new = new.replace("?8:4;", f"?{team * 2}:{team};")
        header = header.replace(old, new)
    return header
