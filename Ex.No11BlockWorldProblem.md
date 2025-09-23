# Ex.No: 11  Planning –  Block World Problem 
### DATE: 23-09-2025                                                                           
### REGISTER NUMBER : 212223040166 
### AIM: 
To find the sequence of plan for Block word problem using PDDL  
###  Algorithm:
Step 1 :  Start the program <br>
Step 2 : Create a domain for Block world Problem <br>
Step 3 :  Create a domain by specifying predicates clear, on table, on, arm-empty, holding. <br>
Step 4 : Specify the actions pickup, putdown, stack and un-stack in Block world problem <br>
Step 5 :  In pickup action, Robot arm pick the block on table. Precondition is Block is on table and no other block on specified block and arm-hand empty.<br>
Step 6:  In putdown action, Robot arm place the block on table. Precondition is robot-arm holding the block.<br>
Step 7 : In un-stack action, Robot arm pick the block on some block. Precondition is Block is on another block and no other block on specified block and arm-hand empty.<br>
Step 8 : In stack action, Robot arm place the block on under block. Precondition is Block holded by robot arm and no other block on under block.<br>
Step 9 : Define a problem for block world problem.<br> 
Step 10 : Obtain the plan for given problem.<br> 
     
### Program:
```
(define (domain blocksworld)
(:requirements :strips :equality)
(:predicates (clear ?x)
(on-table ?x)
(arm-empty)
(holding ?x)
(on ?x ?y))
(:action pickup
:parameters (?ob)
:precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
:effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob))
(not (arm-empty))))
(:action putdown
:parameters (?ob)
:precondition (and (holding ?ob))
:effect (and (clear ?ob) (arm-empty) (on-table ?ob)
(not (holding ?ob))))
(:action stack
:parameters (?ob ?underob)
:precondition (and (clear ?underob) (holding ?ob))
:effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
(not (clear ?underob)) (not (holding ?ob))))
(:action unstack
:parameters (?ob ?underob)
:precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
:effect (and (holding ?ob) (clear ?underob)
(not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))
```








### Input 
Problem1
```
(define (problem pb1)
(:domain blocksworld)
(:objects a b)
(:init (on-table a) (on-table b) (clear a) (clear b) (arm-empty))
(:goal (and (on a b))))
```
Problem2
```
(define(problem pb3)
(:domain blocksworld)
(:objects a b c)
(:init (on-table a) (on-table b) (on-table c)
(clear a) (clear b) (clear c) (arm-empty))
(:goal (and (on a b) (on b c))))
```

### Output/Plan:
### Problem1:

<img width="401" height="527" alt="image" src="https://github.com/user-attachments/assets/be7e1a8e-7d66-4801-91a0-5dd9a7baf4bf" /><br>


<img width="401" height="527" alt="image" src="https://github.com/user-attachments/assets/a0cf7868-558a-4ecb-9477-9e18c997341d" />

### Problem2:

<img width="409" height="523" alt="image" src="https://github.com/user-attachments/assets/ac46be21-fb0d-41d1-90f1-345ed189e9c1" /><br>


<img width="409" height="523" alt="image" src="https://github.com/user-attachments/assets/b72f8613-67bb-4e14-b31e-f3d7fa0c3b80" /><br>
<img width="409" height="523" alt="image" src="https://github.com/user-attachments/assets/9e29ceef-f94f-42a8-ba47-bf1547e3a2c4" /><br>
<img width="409" height="523" alt="image" src="https://github.com/user-attachments/assets/e252ba14-bebf-40ee-8ff3-e9b419324524" />




### Result:
Thus the plan was found for the initial and goal state of block world problem.
