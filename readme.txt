Microsoft Visual C++ Redistributable
pip uninstall tensorflow tensorflow-cpu keras -y
pip install "tensorflow-cpu==2.13.0" "keras==2.13.1"
python train_model.py
python main_app.py