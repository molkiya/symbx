# demo_register_and_run.py
from symbx_db import SymbXDB
from symbx_rules import canonical_form

if __name__ == "__main__":
    db = SymbXDB()  # uses local defaults

    # 1) ensure rules exist
    db.bootstrap_rules()

    # 2) define a program (*3 ; +2 ; *2)
    prog_names = ["*3.0", "+2.0", "*2.0"]  # must match RULES names
    program_id, phash = db.upsert_program(prog_names)
    print("program_id:", program_id, "prog_hash:", phash, "canon:", canonical_form(prog_names))

    # 3) execute with cache
    x_pred, mse, ph = db.exec_with_cache(prog_names, a_value=1.5)
    print("x_pred:", x_pred, "mse:", mse, "hash:", ph)

    # 4) log episode (dummy)
    ep_id = db.log_episode(
        experiment_id=1, task_id=None, program_id=program_id,
        x_pred=x_pred, mse=0.0, reward=0.0, steps_count=len(prog_names),
        trace_uri=None
    )
    print("episode_id:", ep_id)
