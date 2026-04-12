import subprocess, json, sys
import os

print("--- Running test_scores checks ---")
# To run this, we need the background API server running.
# Let's skip the inference.py HTTP call for a moment unless we start the API server.
# Wait, the fallback is to use the local environment directly if USE_LOCAL_ENV=true.
# In the updated inference.py: USE_LOCAL_ENV: bool = os.environ.get("USE_LOCAL_ENV", "true").lower() == "true"
# So it can run entirely locally without the HTTP server!

# Set USE_LOCAL_ENV=true explicitly
env = os.environ.copy()
env["USE_LOCAL_ENV"] = "true"

result = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True, text=True, timeout=300, env=env
)

output = result.stdout
if result.returncode != 0:
    print("inference.py failed!")
    print("Stderr:", result.stderr)

task_scores = []
final_score = None

for line in output.splitlines():
    if line.startswith("[TASK]"):
        data = json.loads(line[7:])
        task_scores.append(data["score"])
        assert 0.0 < data["score"] < 1.0, f"FAIL: task {data['task_id']} score={data['score']} is on boundary!"
    if line.startswith("[END]"):
        data = json.loads(line[6:])
        final_score = data["final_score"]
        assert 0.0 < final_score < 1.0, f"FAIL: final_score={final_score} is on boundary!"

print(f"All {len(task_scores)} task scores pass: {task_scores}")
print(f"Final score: {final_score}")


print("\n--- Running grader boundary checks ---")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.openenv.graders import _safe_clamp, EasyGrader, MediumGrader, HardGrader, ActionRecord

# Test _speed_bonus boundaries
g = EasyGrader()
assert g._speed_bonus(0, 15, 30) < 1.0,  "Speed bonus at step 0 must be < 1.0"
assert g._speed_bonus(30, 15, 30) > 0.0, "Speed bonus at max step must be > 0.0"
assert g._speed_bonus(15, 15, 30) < 1.0, "Speed bonus at optimal must be < 1.0"

# Test _safe_clamp
assert _safe_clamp(0.0) > 0.0
assert _safe_clamp(1.0) < 1.0
assert _safe_clamp(0.5) == 0.5

print("All grader boundary checks passed.")
