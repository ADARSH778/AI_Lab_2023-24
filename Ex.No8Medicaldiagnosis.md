# Ex.No: 8  Logic Programming –  Medical Diagnosis Expert System
### DATE: 13-09-2025                                                                           
### REGISTER NUMBER : 212223040166
### AIM: 
Write a Prolog program to build a medical Diagnosis Expert System.
###  Algorithm:
1. Start the program.
2. Write the rules for each diseases.
3. If patient have mumps then symptoms are fever and swollen glands.
4. If patient have cough, sneeze and running nose then disease is measles.
5. if patient have symptoms headache ,sneezing ,sore_throat, runny_nose and  chills then disease is common cold.
6. Define rules for all disease.
7. Call the predicates and Collect the symptoms of Patient and give the hypothesis of disease.
        

### Program:
```
hypothesis(Patient,german_measles) :-
    symptom(Patient,fever),
    symptom(Patient,headache),
    symptom(Patient,runny_nose),
    symptom(Patient,rash).
hypothesis(Patient,flu) :-
    symptom(Patient,fever),
    symptom(Patient,headache),
    symptom(Patient,body_ache),
    symptom(Patient,conjunctivitis),
    symptom(Patient,chills),
    symptom(Patient,sore_throat),
    symptom(Patient,runny_nose),
    symptom(Patient,cough).
hypothesis(Patient,common_cold) :-
    symptom(Patient,headache),
    symptom(Patient,sneezing),
    symptom(Patient,sore_throat).
hypothesis(Patient,chicken_pox) :-
    symptom(Patient,fever),
    symptom(Patient,chills),
    symptom(Patient,body_ache),
    symptom(Patient,rash).
hypothesis(Patient,measles) :-
    symptom(Patient,cough),
    symptom(Patient,sneezing),
    symptom(Patient,runny_nose).
symptom(raju,headache).
symptom(raju,sneezing).
symptom(raju,sore_throat).
```










### Output:

<img width="933" height="386" alt="image" src="https://github.com/user-attachments/assets/5f65d9cf-ca03-4d65-8140-6428c9c785d2" />


<img width="931" height="357" alt="image" src="https://github.com/user-attachments/assets/d8283ea3-84f8-4424-9531-eaea30e5f2d0" />

<img width="929" height="368" alt="image" src="https://github.com/user-attachments/assets/6c190894-9303-40d6-a579-4fc0228a355d" />

<img width="929" height="351" alt="image" src="https://github.com/user-attachments/assets/ff2f1947-cebd-4e81-a05b-39943fe5d334" />





### Result:
Thus the simple medical diagnosis system was built sucessfully.
