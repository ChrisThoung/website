---
layout: post
title: Python virtual environments on Windows Subsystem for Linux
type: post
hide: false
tags: setup notes linux ubuntu wsl python virtualenv
---

Steps to set up Python virtual environments in [Ubuntu
Linux](https://ubuntu.com/) (20.04 LTS: Focal Fossa) running on [Windows
Subsystem for Linux](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux)
(WSL) for Windows 10. I'm running WSL 2 but I imagine WSL 1 works the same
way. This is also almost certainly fine on an actual Ubuntu machine but I've
been futzing around with [my Linux
laptop](/2017/08/06/ubuntu-1704-setup-ideapad/) so can't test that right now.

This came up as I try to figure out how to get a test suite running (as yet
[unsuccessfully](https://travis-ci.org/github/ChrisThoung/fsic/builds/716384401))
on different versions of Python 3.

1. Add the
   [`deadsnakes`](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)
   Personal Package Archive (PPA) to your `sources.list`:  
   `$ sudo apt-add-repository ppa:deadsnakes/ppa`
2. Install [`virtualenv`](https://virtualenv.pypa.io/en/latest/):  
   `$ python -m pip install virtualenv`
3. Install whichever version(s) of Python you need e.g. Python 3.6, as here:  
   `$ sudo apt install python3.6`
4. Depending on what you're doing (for me, some work involving compilation with
   [F2PY](https://numpy.org/doc/stable/f2py/)), you might need the accompanying
   header files and static library:  
   `$ sudo apt install python3.6-dev`  
   (you don't *need* this to set up the virtual environment and can always
   install it later without needing to redo any of these other steps)
5. Create a new virtual environment with the command:  
   `$ virtualenv -p /usr/bin/python3.6 --clear ~/python/python36`  
   where:
    * the `-p` option points to the Python version
	* `--clear` removes (clears) the destination folder (if found)
	* the last argument, `~/python/python36`, specifies the destination folder

To activate the environment (changing the path to match the destination folder
above as needed):

    $ source ~/python/python36/bin/activate

`pip` etc should work as usual.

To deactivate:

    $ deactivate
