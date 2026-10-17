# RNNs

This directory contains recurrent neural network implementations using NumPy.

- `0-rnn_cell.py`: simple RNN cell with tanh hidden state and softmax output.
- `1-rnn.py`: forward propagation through a sequence of inputs.
- `2-gru_cell.py`: gated recurrent unit cell.
- `3-lstm_cell.py`: long short-term memory cell.
- `4-deep_rnn.py`: forward propagation through stacked RNN cells.
- `5-bi_forward.py`: forward direction of a bidirectional cell.
- `6-bi_backward.py`: backward direction of a bidirectional cell.
- `7-bi_output.py`: output calculation for bidirectional hidden states.
- `8-bi_rnn.py`: complete bidirectional RNN propagation.

All recurrent inputs use time-major arrays with shape `(time, batch, features)`.
