# Ex.No: 7  Logic Programming –  Logic Circuit Design
### DATE: 09-09-2025                                                                           
### REGISTER NUMBER : 212223040166
### AIM: 
To write a logic program to design a circuit like half adder and half subtractor.
###  Algorithm:
1. Start the Program
2. Design a AND gate logic if both inputs are 1 then output is 1.
3. Design a OR gate logic if any one of input is 1 then output is 1.
4. Design a XOR gate logic if both inputs are different then output is 1.
5. Design a NOT gate logic if input is 0 then output is 1.
6. Design a half adder and half subtractor using the rules.
7. Test the logic.
8. Stop the program.

### Program:
```
halfadder(X,Y,Sum,Carry):-xor(X,Y,Sum),and(X,Y,Carry).
halfsub(X,Y,Diff,Borrow):-xor(X,Y,Diff),not(X,Z),and(Z,X,Borrow).
fulladd(X,Y,Z,Sum,Carry):-xor(X,Y,A),xor(A,Z,Sum),and(A,Z,B),and(X,Y,C),or(B,C,Carry).
xor(0,0,0).
xor(0,1,1).
xor(1,0,1).
xor(1,1,0).
and(0,0,0).
and(0,1,0).
and(1,0,0).
and(1,1,1).
or(0,1,1).
or(0,0,0).
or(1,0,1).
or(1,1,1).
not(0,1).
not(1,0).
```










### Output:
## Half Adder:

<img width="936" height="378" alt="image" src="https://github.com/user-attachments/assets/e195a475-0a37-4cba-9570-689b689457a5" />

## Half Subtractor:

<img width="934" height="378" alt="image" src="https://github.com/user-attachments/assets/3d7e8c7f-30ac-438a-8f1a-9c48cd4a203b" />


## Full Adder:

<img width="934" height="379" alt="image" src="https://github.com/user-attachments/assets/4449071e-c4cf-40fe-befb-91d6fa0a5682" />


### Result:
Thus the truth table of circuit verified sucessfully.
