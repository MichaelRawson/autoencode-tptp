# Autoencoding TPTP

This repo supplements the [paper](http://aitp-conference.org/2020/abstract/paper_27.pdf) by Michael Rawson and Giles Reger.
Later work by @inpefess, thanks!

# License
MIT licensed.

# Before running

* install Python dependencies (e.g. `poetry install` should work)
* from `tptp-graph` folder run `maturin build` (for this you'll need Rust installation on your system)
* from project root run `pip install tptp-graph/target/wheel/wheel-of-your-choice.whl`
   * you may find it easier to `cargo build --release` in `/tptp-graph`, followed by symlinking `/tptp-graph/target/release/libtptp_graph.so` to `/tptp_graph.so` for the preceding two steps
* subfolders `Problems` and `Axioms` should be in your working folder (either run scripts from TPTP folder or create symbolic links in the project root)

# Training

`python train.py`

That can take a while. You can edit `data.py` file to use only a subset of TPTP.

Training never ends! Run `tensorboard --logdir runs` to decide when to stop it manually.

After stopping the training script, `mv save.pt model.pt`.

# Getting embeddings

`python embed.py`

This should create a file called `embeddings.pkl` in project root. File includes a dictionary with problem names as keys and embeddings as values.
