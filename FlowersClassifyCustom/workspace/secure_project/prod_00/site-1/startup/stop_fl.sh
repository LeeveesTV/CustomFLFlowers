#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Shutdown request created.  Wait for local FL process to shutdown."
touch $DIR/../shutdown.fl
sleep 5
if [[ ! -f "$DIR/../pid.fl" ]]; then
  echo "No pid.fl.  No need to kill process."
  exit
fi
pid=`cat $DIR/../pid.fl`
kill -0 ${pid} 2> /dev/null 1>&2
if [[ $? -ne 0 ]]; then
  echo "Process already terminated"
  exit
fi
kill -9 $pid
pid=`cat $DIR/../tee.fl`
kill -9 $pid
rm -f $DIR/../pid.fl $DIR/../shutdown.fl $DIR/../restart.fl $DIR/../tee.fl
echo "Shutdown process finished."
