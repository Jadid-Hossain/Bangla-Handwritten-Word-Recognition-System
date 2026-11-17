
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "!! python3 is required (Python 3.10 - 3.12 works with these pins)" >&2
    exit 1
fi

# Ubuntu needs the venv module as a separate package; install it when missing
if ! python3 -m venv /tmp/_venv_probe 2>/dev/null; then
    echo "==> Installing python3-venv"
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update && apt-get install -y python3-venv python3-pip
    else
        sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
    fi
fi
rm -rf /tmp/_venv_probe

echo "==> Creating virtual environment at venv/"
python3 -m venv venv
source venv/bin/activate

echo "==> Installing requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import tensorflow as tf
print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
PY
docker --version || echo "!! docker not found on PATH; install Docker Engine or Docker Desktop"

echo "==> Done. Activate with:  source venv/bin/activate"
echo "    Train:                python train.py"
echo "    App:                  streamlit run app.py"
