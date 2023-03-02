#!/usr/bin/env bash
clingo -W no-atom-undefined -t 1 0 --single-shot --project -c bounded_nonreach=0 "${@}" - <<EOF
#program base.
{clause(N,1..C,L,S): in(L,N,S), maxC(N,C), node(N), node(L)}.
:- clause(N,_,L,S), clause(N,_,L,-S).
1 { constant(N,(-1;1)) } 1 :- node(N), not clause(N,_,_,_).
constant(N) :- constant(N,_).
size(N,C,X) :- X = #count {L,S: clause(N,C,L,S)}; clause(N,C,_,_); maxC(N,_).
:- clause(N,C,_,_); not clause(N,C-1,_,_); C > 1; maxC(N,_).
:- size(N,C1,X1); size(N,C2,X2); X1 < X2; C1 > C2; maxC(N,_).
:- size(N,C1,X); size(N,C2,X); C1 > C2; mindiff(N,C1,C2,L1) ; mindiff(N,C2,C1,L2) ; L1 < L2; maxC(N,_).
clausediff(N,C1,C2,L) :- clause(N,C1,L,_);not clause(N,C2,L,_);clause(N,C2,_,_), C1 != C2; maxC(N,_).
mindiff(N,C1,C2,L) :- clausediff(N,C1,C2,L); L <= L' : clausediff(N,C1,C2,L'), clause(N,C1,L',_), C1!=C2; maxC(N,_).
:- size(N,C1,X1); size(N,C2,X2); C1 != C2; X1 <= X2; clause(N,C2,L,S) : clause(N,C1,L,S); maxC(N,_).
edge(L,N,S) :- clause(N,_,L,S).
nbnode(6).
node("BDNF").
node("ASCL1").
node("EDN3").
node("GDNF").
node("PHOX2B").
node("RET").
in("BDNF","ASCL1",1).
in("BDNF","EDN3",1).
in("BDNF","GDNF",1).
in("BDNF","PHOX2B",1).
in("ASCL1","BDNF",1).
in("ASCL1","EDN3",1).
in("ASCL1","GDNF",1).
in("ASCL1","PHOX2B",1).
in("ASCL1","RET",1).
in("EDN3","ASCL1",1).
in("EDN3","BDNF",1).
in("EDN3","GDNF",1).
in("EDN3","PHOX2B",1).
in("EDN3","RET",1).
in("GDNF","ASCL1",1).
in("GDNF","BDNF",1).
in("GDNF","EDN3",1).
in("GDNF","PHOX2B",1).
in("PHOX2B","ASCL1",1).
in("PHOX2B","BDNF",1).
in("PHOX2B","EDN3",1).
in("PHOX2B","GDNF",1).
in("PHOX2B","RET",1).
in("RET","ASCL1",1).
in("RET","EDN3",1).
in("RET","PHOX2B",1).
maxC("BDNF",3).
maxC("ASCL1",3).
maxC("EDN3",3).
maxC("GDNF",3).
maxC("PHOX2B",3).
maxC("RET",3).
:- in(L,N,S), not edge(L,N,S).
#show clause/4.
#show constant/2.

EOF
