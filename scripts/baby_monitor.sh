#!/bin/bash

PID=""
WORKDIR="/home/dakre/work/baby_monitor"

cd $WORKDIR

function get_pid {
    PID=`pidof venv/bin/python baby_monitor.py`
}

function start {
    get_pid

    echo  "Starting server.."
    venv/bin/python baby_monitor.py &
    get_pid

    echo "Done. PID=$PID"
}

start
