% Stanisław Bitner 438247

:- ensure_loaded(library(lists)).
:- op(700, xfx, <>).

% verify(+ProcN, +Prog)
% ProcN - liczba procesów
% ProgFile - nazwa pliku z programem
% Sprawdza, czy program jest bezpieczny, jeśli nie, to wypisuje błędny przeplot
% oraz procesy znajdujące się w sekcji krytycznej, po wykonaniu tego przeplotu.
verify(ProcN, ProgFile) :-
    (
        number(ProcN), ProcN > 0 -> true;
        format('Error: parametr ~w powinien byc liczba > 0~n', [ProcN]), fail
    ),
    catch(readFileAndRun(ProcN, ProgFile), _, (
        format("Error: brak pliku o nazwie - ~w", ProgFile), fail
    )), !.

% readFileAndRun(+ProcN, +ProgFile)
% ProcN - liczba procesów
% ProgFile - nazwa pliku z programem
% Wczytuje program z pliku i sprawdza jego poprawność.
readFileAndRun(ProcN, ProgFile) :-
    see(ProgFile),
    !,
    read(variables(Variables)),
    read(arrays(Arrays)),
    read(program(Prog)),
    seen,
    run(ProcN, Variables, Arrays, Prog).

% run(+ProcN)
% ProcN - liczba procesów
% Variables - zmienne
% Arrays - tablice
% Prog - program (lista instrukcji)
% Uruchamia program dla ProcN procesów i wypisuje wynik.
run(ProcN, Variables, Arrays, Prog) :-
    initState(prog(Variables, Arrays, Prog), ProcN, State),
    search(Prog, [(0,State,[])], [], ProcN, [], [], Result),
    printResultult(Prog, Result).

% {{{ Print
% printResultult(+Prog, +Result)
% Prog - program
% Result - wynik działania programu
printResultult(_, []) :-
    write('Program jest poprawny (bezpieczny).\n').
printResultult(Prog, result(Path, State)) :-
    write('Program jest niepoprawny.\n'),
    write('Niepoprawny przeplot:\n'),
    printPath(Path),
    write('Procesy w sekcji:'),
    getProcessesInSection(Prog, State, Procs),
    reverse(Procs, [Fp|RevProcs]),
    format(' ~d', [Fp]),
    printProcesses(RevProcs),
    write('.\n').

% printPath(+Path)
% Path - ścieżka wykonania programu
% Wypisuje niepoprawny przeplot.
printPath([]).
printPath([move(Pid, Position)|Rest]) :-
    printPath(Rest),
    format('  Proces ~d: ~d~n', [Pid, Position]).

% printProcesses(+Procs)
% Procs - lista procesów w sekcji
% Wypisuje procesy w sekcji.
printProcesses([]).
printProcesses([Pid|Rest]) :-
    printProcesses(Rest),
    format(', ~d', [Pid]).

% getProcessesInSection(+Prog, +State, -Procs)
% Prog - program
% State - stan
% Procs - lista procesów w sekcji
% Zwraca listę procesów w sekcji.
getProcessesInSection(Prog, state(_, _, Positions), Procs) :-
    getProcessesInSection(Prog, Positions, 0, [], Procs).
getProcessesInSection(_, [], _, Acc, Acc).
getProcessesInSection(Prog, [Position|Rest], Pid, Acc, Procs) :-
    NewPid is Pid + 1,
    (
        nth1(Position, Prog, sekcja) ->
        getProcessesInSection(Prog, Rest, NewPid, [Pid|Acc], Procs);
        getProcessesInSection(Prog, Rest, NewPid, Acc, Procs)
    ).
% }}} Print
% {{{ Init
% initState(+Variables, +Arrays, +ProcN, -State)
% Variables - zmienne
% Arrays - tablice
% ProcN - liczba procesów
% State - stan początkowy
% Inicjalizuje stan początkowy.
% Zmienne i tablice są inicjalizowane na 0.
initState(prog(Variables, Arrays, _), ProcN,
          state(VarValues, ArrValues, Positions)) :-
    maplist(initVar,        Variables, VarValues),
    maplist(initArr(ProcN), Arrays,    ArrValues),
    initList(ProcN, 1, Positions).

% initVar(+Name, -Var)
% Name - nazwa zmiennej
% Var - zmienna
% Inicjalizuje zmienną na 0.
initVar(Name, vvar(Name, 0)).

% initArr(+ProcN, +Name, -Arr)
% ProcN - liczba procesów
% Name - nazwa tablicy
% Arr - tablica
% Inicjalizuje tablicę na 0.
initArr(ProcN, Name, varr(Name, EmptyArr)) :-
    initList(ProcN, 0, EmptyArr).

% initList(+Size, +Val, -List)
% Size - rozmiar listy
% Val - wartość elementów listy
% List - lista
% Inicjalizuje listę na wartość Val.
initList(Size, Val, List) :-
    initList(Size, Val, [], List).
initList(0, _, Acc, Acc).
initList(Size, Val, Acc, List) :-
    NewSize is Size - 1,
    initList(NewSize, Val, [Val|Acc], List).
% }}} Init
% {{{ Search
% search(+Prog, +Layer, +ProcN, +Visited, +ResultIn, -ResultOut)
% Prog - program
% Layer - warstwa
% ProcN - liczba procesów
% Visited - lista odwiedzonych wierzchołków
% ResultIn - wynik
% ResultOut - wynik
% Wyszukuje niepoprawny przeplot. Jeśli nie znajdzie, to zwraca pustą listę,
% jeśli znajdzie, to zwraca wynik postaci result(Path, State).
search(_, [], [], _, _, [], []).

search(Prog, [(Pid,State,Path)|States], Layer, ProcN, Visited, [], ResultOut) :-
    stateIsSafe(State, Prog),
    stateGetPos(State, Pid, Position),
    step(prog(_, _, Prog), State, Pid, NextState),
    NextPid is Pid + 1,
    NextNode = (0,NextState,[move(Pid,Position)|Path]),
    addNode((NextPid,State,Path), ProcN, Visited, Layer, Visited1, Layer1),
    addNode(NextNode, ProcN, Visited1, Layer1, Visited2, Layer2),
    search(Prog, States, Layer2, ProcN, Visited2, [], ResultOut).

search(Prog, [(_,State,Path)|_], _, _, _, [], result(Path, State)) :-
    \+ stateIsSafe(State, Prog).

search(Prog, [], Layer, ProcN, Visited, [], ResultOut) :-
    search(Prog, Layer, [], ProcN, Visited, [], ResultOut).

search(_, _, _, _, _, ResultIn, ResultIn).

% addNode(+Node, +ProcN, +Visited, +Layer, -VisitedOut, -LayerOut)
% Node - wierzchołek
% ProcN - liczba procesów
% Visited - lista odwiedzonych wierzchołków
% Layer - warstwa
% VisitedOut - lista odwiedzonych wierzchołków po dodaniu wierzchołka
% LayerOut - warstwa po dodaniu wierzchołka
% Dodaje wierzchołek do warstwy, jeśli ma poprawny Pid i nie był odwiedzony.
addNode((0,State,Path), _, Visited, Layer,
        [State|Visited], [(0,State,Path)|Layer]) :-
    \+ member(State, Visited).
addNode((Pid,State,Path), ProcN, Visited, Layer,
        Visited, [(Pid,State,Path)|Layer]) :-
    Pid > 0, Pid < ProcN.
addNode(_, _, Visited, Layer, Visited, Layer).
% }}} Search
% {{{ Step
% step(+Prog, +StateIn, +Pid, -StateOut)
% Prog - program
% StateIn - stan wejściowy
% Pid - identyfikator procesu
% StateOut - stan wyjściowy
% Wykonuje krok programu.
step(prog(_, _, Prog), StateIn, Pid, StateOut) :-
    stateGetPos(StateIn, Pid, Position),
    nth1(Position, Prog, Instr),
    instruction(Instr, Pid, StateIn, StateOut).
% {{{ Instruction
% instruction(+Instr, +Pid, +StateIn, -StateOut)
% Instr - instrukcja
% Pid - identyfikator procesu
% StateIn - stan wejściowy
% StateOut - stan wyjściowy
% Wykonuje instrukcję.
instruction(assign(array(ArrayName, ArrayIndex), Expr),
            Pid, StateIn, StateOut) :-
    evalArithmetic(Expr, StateIn, Pid, Val),
    evalArithmetic(ArrayIndex, StateIn, Pid, IndexVal),
    stateUpdArr(ArrayName, IndexVal, Val, StateIn, StateTmp),
    stateAdvancePosition(StateTmp, Pid, StateOut).

instruction(assign(VarName, Expr),
            Pid, StateIn, StateOut) :-
    evalArithmetic(Expr, StateIn, Pid, Val),
    stateUpdVar(VarName, Val, StateIn, StateTmp),
    stateAdvancePosition(StateTmp, Pid, StateOut).

instruction(goto(NewPosition),
            Pid, StateIn, StateOut) :-
    evalArithmetic(NewPosition, StateIn, Pid, NewPositionVal),
    stateUpdPos(NewPositionVal, StateIn, Pid, StateOut).

instruction(condGoto(LogExpr, NewPosition),
            Pid, StateIn, StateOut) :-
    evalLogic(LogExpr, StateIn, Pid, LogValue),
    LogValue,
    stateUpdPos(NewPosition, StateIn, Pid, StateOut);
    stateAdvancePosition(StateIn, Pid, StateOut).

instruction(sekcja,
            Pid, StateIn, StateOut) :-
    stateAdvancePosition(StateIn, Pid, StateOut).
% }}} Instruction
% }}} Step
% {{{ Eval
% evalArithmetic(+Expr, +State, +Pid, -Value)
% Expr - wyrażenie arytmetyczne
% State - stan
% Pid - identyfikator procesu
% Value - wartość wyrażenia
% Oblicza wartość wyrażenia arytmetycznego.
evalArithmetic(Expr1 + Expr2, State, Pid, Value) :-
    evalArithmetic(Expr1, State, Pid, Val1),
    evalArithmetic(Expr2, State, Pid, Val2),
    Value is Val1 + Val2.
evalArithmetic(Expr1 - Expr2, State, Pid, Value) :-
    evalArithmetic(Expr1, State, Pid, Val1),
    evalArithmetic(Expr2, State, Pid, Val2),
    Value is Val1 - Val2.
evalArithmetic(Expr1 * Expr2, State, Pid, Value) :-
    evalArithmetic(Expr1, State, Pid, Val1),
    evalArithmetic(Expr2, State, Pid, Val2),
    Value is Val1 * Val2.
evalArithmetic(Expr1 / Expr2, State, Pid, Value) :-
    evalArithmetic(Expr1, State, Pid, Val1),
    evalArithmetic(Expr2, State, Pid, Val2),
    Value is Val1 // Val2.
evalArithmetic(Expr, State, Pid, Value) :-
    evalSimple(Expr, State, Pid, Value).

% evalSimple(+Expr, +State, +Pid, -Value)
% Expr - wyrażenie
% State - stan
% Pid - identyfikator procesu
% Value - wartość wyrażenia
% Oblicza wartość prostego wyrażenia.
evalSimple(Number, _, _, Number) :-
    number(Number).
evalSimple(array(Ident, Expr), State, Pid, Value) :-
    evalArithmetic(Expr, State, Pid, Index),
    stateGetArr(Ident, Index, State, Value).
evalSimple(pid, _, Pid, Pid).
evalSimple(Ident, State, _, Value) :-
    \+ number(Ident),
    Ident \= pid,
    stateGetVar(Ident, State, Value).

% evalLogic(+Expr, +State, +Pid, -Value)
% Expr - wyrażenie logiczne
% State - stan
% Pid - identyfikator procesu
% Value - wartość wyrażenia
% Oblicza wartość wyrażenia logicznego.
evalLogic(Expr1 < Expr2, State, Pid, true) :-
    evalArithmetic(Expr1, State, Pid, Val1),
    evalArithmetic(Expr2, State, Pid, Val2),
    Val1 < Val2.
evalLogic(Expr1 = Expr2, State, Pid, true) :-
    evalArithmetic(Expr1, State, Pid, Val1),
    evalArithmetic(Expr2, State, Pid, Val2),
    Val1 =:= Val2.
evalLogic(Expr1 <> Expr2, State, Pid, true) :-
    evalArithmetic(Expr1, State, Pid, Val1),
    evalArithmetic(Expr2, State, Pid, Val2),
    Val1 =\= Val2.
evalLogic(_, _, _, false).
% }}} Eval
% {{{ State
% Opis Stanu:
% state(VarValues, ArrValues, Positions)
% VarValues - lista par vvar(nazwa zmiennej, wartość)
% ArrValues - lista par varr(nazwa tablicy, wartość)
% Positions - lista pozycji procesów (numer następnej wykonywanej instrukcji)

% stateGetPos(+State, +Pid, -Position)
% State - stan
% Pid - identyfikator procesu
% Position - pozycja procesu
% Zwraca pozycję procesu. (numer instrukcji liczony od 1)
stateGetPos(state(_, _, Positions), Pid, Position) :-
    nth0(Pid, Positions, Position).

% stateGetVar(+Name, +State, -Val)
% Name - nazwa zmiennej
% State - stan
% Val - wartość zmiennej
% Zwraca wartość zmiennej.
stateGetVar(Name, state(VarValues, _, _), Val) :- 
    stateGetVar(Name, VarValues, Val).
stateGetVar(Name, [vvar(Name, Val)|_], Val).
stateGetVar(Name, [vvar(_,_)|Resultt], Val) :-
    stateGetVar(Name, Resultt, Val).

% stateGetArr(+Name, +Index, +State, -Val)
% Name - nazwa tablicy
% Index - indeks
% State - stan
% Val - wartość tablicy
% Zwraca wartość elementu tablicy.
stateGetArr(Name, Index, state(_, ArrValues, _), Val) :-
    stateGetArr(Name, Index, ArrValues, Val).
stateGetArr(Name, Index, [varr(Name, ArrVal)|_], Val) :-
    nth0(Index, ArrVal, Val).
stateGetArr(Name, Index, [varr(_,_)|Resultt], Val) :-
    stateGetArr(Name, Index, Resultt, Val).

% stateUpdPos(+NewPosition, +StateIn, +Pid, -StateOut)
% NewPosition - nowa pozycja
% StateIn - stan wejściowy
% Pid - identyfikator procesu
% StateOut - stan wyjściowy
% Aktualizuje pozycję procesu.
stateUpdPos(NewPosition, state(VarValues, ArrValues, Positions), Pid,
            state(VarValues, ArrValues, NewPositions)) :-
    replaceNth0(Positions, Pid, NewPosition, NewPositions).

% stateUpdVar(+Name, +Value, +StateIn, -StateOut)
% Name - nazwa zmiennej
% Value - wartość
% StateIn - stan wejściowy
% StateOut - stan wyjściowy
% Aktualizuje wartość zmiennej.
stateUpdVar(Name, Value,
            state(VarValues, ArrValues, Indexes),
            state(NewVarValues, ArrValues, Indexes)) :-
    getVarIndex(Name, 0, VarValues, Index),
    replaceNth0(VarValues, Index, vvar(Name, Value), NewVarValues).

getVarIndex(Name, CurrentIndex, [vvar(Name, _)|_], CurrentIndex).
getVarIndex(Name, CurrentIndex, [vvar(_, _)|List], Index) :-
    NextIndex is CurrentIndex + 1,
    getVarIndex(Name, NextIndex, List, Index).

% stateUpdArr(+Name, +Index, +Value, +StateIn, -StateOut)
% Name - nazwa tablicy
% Index - indeks
% Value - wartość
% StateIn - stan wejściowy
% StateOut - stan wyjściowy
% Aktualizuje wartość elementu tablicy.
stateUpdArr(Name, Index, Value,
            state(VarValues, ArrValues, Indexes),
            state(VarValues, NewArrValues, Indexes)) :-
    getArrayIndex(Name, ArrValues, ArrayIndex),
    nth0(ArrayIndex, ArrValues, CurrentArray),
    stateUpdArr(Name, Index, Value, CurrentArray, NewArray),
    replaceNth0(ArrValues, ArrayIndex, NewArray, NewArrValues).
stateUpdArr(Name, Index, Value, varr(_, CurrentArray),
            varr(Name, NewArray)) :-
    replaceNth0(CurrentArray, Index, Value, NewArray).

getArrayIndex(Name, ArrValues, Index) :-
    getArrayIndex(Name, 0, ArrValues, Index).
getArrayIndex(Name, CurrentIndex, [varr(Name, _)|_], CurrentIndex).
getArrayIndex(Name, CurrentIndex, [varr(_, _)|Rest], Index) :-
    NextIndex is CurrentIndex + 1,
    getArrayIndex(Name, NextIndex, Rest, Index).

% stateAdvancePosition(+StateIn, +Pid, -StateOut)
% StateIn - stan wejściowy
% Pid - identyfikator procesu
% StateOut - stan wyjściowy
% Przesuwa proces na następną instrukcję.
stateAdvancePosition(StateIn, Pid, StateOut) :-
    stateGetPos(StateIn, Pid, Position),
    NewPosition is Position + 1,
    stateUpdPos(NewPosition, StateIn, Pid, StateOut).

% stateIsSafe(+State, +Prog)
% State - stan
% Prog - program
% Sprawdza czy stan jest bezpieczny, czyli czy w sekcji jest co najwyżej
% jeden proces.
stateIsSafe(state(_, _, Positions), Prog) :-
    countSekcja(Positions, Prog, 0, Count),
    Count < 2.

% countSekcja(+Positions, +Prog, +Acc, -Count)
% Positions - lista pozycji procesów
% Prog - program
% Acc - akumulator
% Count - liczba procesów w sekcji
% Zlicza procesy w sekcji.
countSekcja([], _, Acc, Acc).
countSekcja([Position|Rest], Prog, Acc, Count) :-
    (
        nth1(Position, Prog, sekcja) ->
        NewAcc is Acc + 1,
        countSekcja(Rest, Prog, NewAcc, Count);
        countSekcja(Rest, Prog, Acc, Count)
    ).
% }}} State
% {{{ Utils
% https://stackoverflow.com/questions/61028457/
replaceNth0(List, Index, NewElem, NewList) :-
    nth0(Index, List, _, Transfer),
    nth0(Index, NewList, NewElem, Transfer).
% }}} Utils
