#!/bin/bash
#starts the web app for a demonstration.
#
#run it from anywhere:  bash "START.sh"
#then open http://<this machine>:5000 in a browser.
#
#the two OPENBLAS lines are not cosmetic.  this board has four fast cores and
#two slow ones, and every matrix block waits for the slowest thread, so
#letting the library take all six makes a page take 54 s instead of 44 s.
#four threads keeps the work on the fast cores.
cd "$(dirname "$0")" || exit 1

export OPENBLAS_NUM_THREADS=4
export OPENBLAS_CORETYPE=ARMV8

#which reader checkpoint to use.  recognise.py names the scratch one, which
#reads about a quarter of the characters; the joint one is the checkpoint the
#reported accuracy was measured on.  comment the line out to use the default.
export READER_WEIGHTS=paper_cnn_bilstm_ctc_joint_best.npz

echo "reader   : ${READER_WEIGHTS:-the default named in recognise.py}"
echo "threads  : $OPENBLAS_NUM_THREADS"
echo "open      http://$(hostname -I | awk '{print $1}'):5000"
echo

exec python3 web/server.py
