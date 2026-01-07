#!/bin/bash

echo "=============================================="
echo "Copying over lab code to the my-work directory"
echo "=============================================="

pushd . > /dev/null

cd ~/morf-synt-2026/

for d in Lab*; do
  echo "Found $d"
  if [[ -e ~/my-work/$d ]]
  then
      echo "$d has already been copied to my-work. Doing nothing about that one."
  else
      echo "$d does not yet exist in my-work. Creating a copy."
      cp -r $d ~/my-work
  fi
done

echo "Found no more labs to copy."

popd > /dev/null

