set -x

git submodule update --init --recursive --force
python3 -m pip uninstall toy-hgemm -y
python3 setup.py bdist_wheel && cd dist && python3 -m pip install *.whl && cd -
rm -rf toy_hgemm.egg-info __pycache__

set +x