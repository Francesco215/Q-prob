sudo apt update
sudo apt install pip ffmpeg tmux btop -y

pip install uv nvitop swig

uv venv 
. .venv/bin/activate
uv pip install -e .