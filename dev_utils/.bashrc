#!/bin/bash

# Git Workflow Shortcuts
alias gs="git status"
alias gd="git diff"
alias gb="git branch"
alias gc="git checkout"
alias ga="git add"
alias gcm="git commit -m"
alias gca="git commit --amend"
alias gpo="git push origin"
alias gp="git pull"
alias gpr="git pull --rebase"
alias glog="git log --oneline --graph --decorate"
alias gst="git stash"
alias gstp="git stash pop"
alias d="git checkout develop"
alias m="git checkout master"
alias gmd="git merge develop"
alias ls="ls -alh --color=auto"
alias venv="source venv/bin/activate"
alias cdpc="cd Projects/currency_intelligence/"
alias cdpcv="cdpc && venv"
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

# Customize the prompt (PS1)
# \u = username, \h = hostname, \w = working directory
# \[$(tput setaf 6)\] = Cyan color for the branch
export PS1="\[\e[32m\]\u@\h\[\e[m\]:\[\e[34m\]\w\[\e[m\] \[\e[36m\]\$(parse_git_branch)\[\e[m\]\$ "
