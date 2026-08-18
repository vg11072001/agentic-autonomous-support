#!/bin/bash

set -e

# Load .env variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env file not found"
    exit 1
fi
echo "$CRDB_URL"
echo "$AWS_REGION"

echo "=========================================="
echo " Starting Autonomous Support Pipeline"
echo "=========================================="

PYTHON="${PYTHON:-python}"

run_step() {
    echo ""
    echo "=========================================="
    echo "==> Running: $*"
    echo "=========================================="

    "$@"

    echo "==> Completed: $*"
}

# ------------------------------------------
# 1. Generate tabular data
# ------------------------------------------
run_step "$PYTHON" "backend/data_scripts/generate_tabular.py"

# ------------------------------------------
# 2. Generate KB content
# ------------------------------------------
run_step "$PYTHON" "backend/data_scripts/generate_kb_content.py"
run_step "$PYTHON" "backend/data_scripts/ingest_public_kb.py"

# ------------------------------------------
# 3. Generate conversations
# ------------------------------------------
run_step "$PYTHON" "backend/data_scripts/generate_conversations.py"
# run_step "$PYTHON" "backend/data_scripts/backfill_agent_turns.py"

# ------------------------------------------
# 4. Validation test suite
# ------------------------------------------
run_step "$PYTHON" "backend/data_scripts/validation_test_suite.py"

# ------------------------------------------
# 5. Load tabular data into CockroachDB
# ------------------------------------------
run_step "$PYTHON" "backend/to_crdb/load_tabular_to_crdb.py" "--reset-schema"

# ------------------------------------------
# 6. Chunk and embed KB
# ------------------------------------------
run_step "$PYTHON" "backend/to_crdb/chunk_and_embed_kb.py"

# ------------------------------------------
# 7. Load conversations into CockroachDB
# ------------------------------------------
run_step "$PYTHON" "backend/to_crdb/load_conversations_to_crdb.py"

# ------------------------------------------
# Load skills into CockroachDB
# ------------------------------------------
run_step "$PYTHON" "backend/agent/skills.py"


# ------------------------------------------
# 8. Test orchestrator
# ------------------------------------------
run_step "$PYTHON" "backend/agent/orchestrator.py" "233896de79986082f1f479f1f85281cb" \
    "Can I get a refund for my last order?"

# ------------------------------------------
# 9. Guardrail test
# ------------------------------------------
run_step "$PYTHON" "backend/agent/guardrail.py"

# ------------------------------------------
# 10. Simulation flywheel
# ------------------------------------------
run_step "$PYTHON" \
    "backend/agent/simulation_flywheel.py" \
    "--n" "1" \
    "--prompt-version" "v2"

echo ""
echo "=========================================="
echo " Pipeline completed successfully."
echo " Starting API server..."
echo "=========================================="

exec "$PYTHON" -m uvicorn backend.agent.api_server:app \
    --host 0.0.0.0 \
    --port 8000